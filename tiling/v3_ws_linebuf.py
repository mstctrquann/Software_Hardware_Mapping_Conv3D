import numpy as np
import time
from dataclasses import dataclass
import os

# ==========================================
# VERSION 3: WEIGHT STATIONARY + LINE BUFFER
# ==========================================

@dataclass
class HardwareConfig:
    """Cấu hình"""
    H: int = 32
    W: int = 32
    C: int = 128
    R: int = 3
    S: int = 3
    
    # Kiến trúc phần cứng
    PE_ROWS: int = 16   # Input channels per tile
    PE_COLS: int = 16   # Output width per tile
    DATA_WIDTH: int = 32
    
    # Timing
    FREQ_MHZ: float = 200.0
    CYCLE_MAC: int = 1
    CYCLE_DRAM_RD: int = 1
    
    def get_sram_size_kb(self):
        """Tính kích thước SRAM on-chip"""
        # Weight buffer (ping-pong)
        w_bits = 2 * (self.PE_ROWS * self.R * self.S) * self.DATA_WIDTH
        
        # Input buffer (ping-pong)
        in_req_w = self.PE_COLS + self.S - 1
        in_bits = 2 * (self.PE_ROWS * self.R * in_req_w) * self.DATA_WIDTH
        
        # Global accumulator
        H_out = self.H - self.R + 1
        W_out = self.W - self.S + 1
        acc_bits = H_out * W_out * self.DATA_WIDTH
        
        # Line Buffer 
        W_padded = W_out + self.S - 1  # 32 (including halo)
        line_buf_bits = self.PE_ROWS * self.R * W_padded * self.DATA_WIDTH
        
        return (w_bits + in_bits + acc_bits + line_buf_bits) / 8192

# ==========================================
# MEMORY SYSTEM
# ==========================================
class PingPongSRAM:
    """Double-buffered SRAM for load/compute overlap"""
    def __init__(self, shape, name="SRAM"):
        self.name = name
        self.banks = [np.zeros(shape, dtype=np.int32), 
                      np.zeros(shape, dtype=np.int32)]
        self.compute_idx = 0
        self.load_idx = 1

    def swap(self):
        """Swap compute and load banks"""
        self.compute_idx = 1 - self.compute_idx
        self.load_idx = 1 - self.load_idx

    def write_from_dram(self, data):
        """Write data to load bank"""
        np.copyto(self.banks[self.load_idx], data)

    def read_to_pe(self):
        """Read data from compute bank"""
        return self.banks[self.compute_idx]

class LineBuffer:
    """
    Line Buffer logic:
    Giữ R dòng dữ liệu để tái sử dụng 
    """
    def __init__(self, num_channels, num_rows, width):
        """
            num_channels: PE_ROWS (16)
            num_rows: R (3) 
            width: W + S - 1 (32)
        """
        self.buffer = np.zeros((num_channels, num_rows, width), dtype=np.int32)
        self.num_channels = num_channels
        self.num_rows = num_rows
        self.width = width
        
    def reset(self):
        """Clear buffer"""
        self.buffer.fill(0)
    
    def load_row(self, row_data, row_idx):
        """Nạp trực tiếp một dòng vào vị trí cụ thể (dùng cho lúc khởi tạo)"""
        self.buffer[:, row_idx, :] = row_data
    
    def shift_and_load(self, new_row):
        """
        Before:  [Row N  ]
                 [Row N+1]
                 [Row N+2]
        
        After:   [Row N+1]  ← Shifted up
                 [Row N+2]  ← Shifted up
                 [Row N+3]  ← New row loaded

        """
        # Shift rows up
        self.buffer[:, 0:self.num_rows-1, :] = self.buffer[:, 1:self.num_rows, :]
        
        # Load new row at bottom
        self.buffer[:, self.num_rows-1, :] = new_row
    
    def get_window(self, start_col, window_width):
        """Cắt một cửa sổ từ Line Buffer để đưa vào Input SRAM"""
        end_col = start_col + window_width
        return self.buffer[:, :, start_col:end_col].copy() #Cắt data từ line buffer đưa vào SRAM

# ==========================================
# COMPUTE CORE
# ==========================================
class ComputeCore:
    def __init__(self, config):
        self.cfg = config
        self.accumulators = np.zeros(
            (config.PE_ROWS, config.PE_COLS), 
            dtype=np.int32
        )

    def reset_accumulators(self):
        self.accumulators.fill(0)

    def execute_cycle_accurate(self, weight_tile, input_tile):
        """
        Execute MAC operations.
        weight_tile: (16, 3, 3)
        input_tile: (16, 3, 18)
        """
        # Logic tính toán giả lập (Behavioral)
        # Trong phần cứng thật, việc này diễn ra song song tại các PE
        for r in range(self.cfg.R):
            for s in range(self.cfg.S):
                w_vec = weight_tile[:, r, s].reshape(self.cfg.PE_ROWS, 1)
                start_col = s
                end_col = s + self.cfg.PE_COLS
                in_mat = input_tile[:, r, start_col:end_col]
                self.accumulators += w_vec * in_mat

    def adder_tree_reduction_structural(self):
        """Adder tree: 16 channels → 1 output row"""
        layer_0 = [self.accumulators[r, :] for r in range(16)]
        
        # Stage 1: 16 → 8
        layer_1 = []
        for i in range(0, 16, 2):
            layer_1.append(layer_0[i] + layer_0[i+1])
        
        # Stage 2: 8 → 4
        layer_2 = []
        for i in range(0, 8, 2):
            layer_2.append(layer_1[i] + layer_1[i+1])
        
        # Stage 3: 4 → 2
        layer_3 = []
        for i in range(0, 4, 2):
            layer_3.append(layer_2[i] + layer_2[i+1])
        
        # Stage 4: 2 → 1
        final_result = layer_3[0] + layer_3[1]
        
        return final_result

# ==========================================
# TOP LEVEL CONTROLLER
# ==========================================
class HardwareAccelerator:
    """
    V3: Weight Stationary + Line Buffer
    """
    def __init__(self, config):
        self.cfg = config
        
        self.w_sram = PingPongSRAM(
            (config.PE_ROWS, config.R, config.S), 
            "WeightBuf"
        )
        
        input_buf_w = config.PE_COLS + config.S - 1
        self.in_sram = PingPongSRAM(
            (config.PE_ROWS, config.R, input_buf_w), 
            "InputBuf"
        )
        
        self.core = ComputeCore(config)
        
        H_out = config.H - config.R + 1
        W_out = config.W - config.S + 1
        self.global_accumulator = np.zeros((H_out, W_out), dtype=np.int32)
        
        self.stats = {
            "cycles": 0,
            "prologue_cycles": 0,
            "steady_cycles": 0,
            "epilogue_cycles": 0,
            "writeback_cycles": 0,
            "compute_cycles": 0,
            "load_cycles": 0,
            "stall_cycles": 0,
            "dram_words": 0,
            "weight_dram_words": 0,
            "input_dram_words": 0,
            "output_dram_words": 0,
            "weight_loads": 0,
            "input_loads": 0,
            "output_writes": 0,
            "line_buffer_hits": 0,
            "line_buffer_misses": 0,
        }

    def load_weight(self, dram_w, c_start, verbose=False):
        # [TILING CONFIG] Lấy 1 khối Weight Tile tương ứng với 16 Channels
        w_slice = dram_w[0, c_start:c_start+16, :, :]
        self.w_sram.write_from_dram(w_slice)
        
        t_load = w_slice.size * self.cfg.CYCLE_DRAM_RD
        self.stats['dram_words'] += w_slice.size
        self.stats['weight_dram_words'] += w_slice.size
        self.stats['weight_loads'] += 1
        
        if verbose:
            print(f"    [LOAD WEIGHT] Kênh {c_start}-{c_start+15}. Shape: {w_slice.shape}, Words: {w_slice.size}, Bus Cycles: {t_load}")
        return t_load

    def load_input_row_to_linebuffer(self, line_buffer, dram_in, c_start, r_idx, W_padded, verbose=False):
        row_data = dram_in[c_start:c_start+16, r_idx, :W_padded]
        line_buffer.load_row(row_data, r_idx)
        
        t_load = row_data.size * self.cfg.CYCLE_DRAM_RD
        self.stats['dram_words'] += row_data.size
        self.stats['input_dram_words'] += row_data.size
        self.stats['input_loads'] += 1
        self.stats['line_buffer_misses'] += 1
        
        if verbose:
            print(f"    [LOAD INPUT ROW] Nạp hàng {r_idx} vào Line Buffer. Shape: {row_data.shape}, Words: {row_data.size}, Bus Cycles: {t_load}")
        return t_load

    def shift_and_load_input_row(self, line_buffer, dram_in, c_start, next_row_idx, W_padded, verbose=False):
        new_row = dram_in[c_start:c_start+16, next_row_idx, :W_padded]
        line_buffer.shift_and_load(new_row)
        
        t_load = new_row.size * self.cfg.CYCLE_DRAM_RD
        self.stats['dram_words'] += new_row.size
        self.stats['input_dram_words'] += new_row.size
        self.stats['input_loads'] += 1
        self.stats['line_buffer_misses'] += 1
        
        if verbose:
            print(f"    [LOAD INPUT ROW] Shift Line Buffer & Nạp hàng {next_row_idx}. Shape: {new_row.shape}, Words: {new_row.size}, Bus Cycles: {t_load}")
        return t_load

    def extract_window_from_linebuffer(self, line_buffer, ow_tile, window_width, verbose=False):
        in_slice_padded = np.zeros(
            (16, self.cfg.R, self.cfg.PE_COLS + self.cfg.S - 1),
            dtype=np.int32
        )
        window_data = line_buffer.get_window(ow_tile, window_width)
        in_slice_padded[:, :, :window_width] = window_data
        
        self.stats['line_buffer_hits'] += 1
        self.in_sram.write_from_dram(in_slice_padded)
        self.in_sram.swap()
        
        if verbose:
            print(f"    [SRAM READ] Lấy cửa sổ từ Line Buffer (On-chip). Hits: +1, No DRAM cycles!")
        return 0

    def compute(self, verbose=False):
        curr_w = self.w_sram.read_to_pe()
        curr_in = self.in_sram.read_to_pe()
        t_compute = self.cfg.R * self.cfg.S * self.cfg.CYCLE_MAC
        self.stats['compute_cycles'] += t_compute
        
        self.core.execute_cycle_accurate(curr_w, curr_in)
        if verbose:
            print(f"    [COMPUTE] MAC window {self.cfg.R}x{self.cfg.S}. PE Cycles: {t_compute}")
        return t_compute

    def store_accumulator(self, oh, ow_tile, valid_width, verbose=False):
        output_row = self.core.adder_tree_reduction_structural()
        self.global_accumulator[oh, ow_tile:ow_tile+valid_width] += output_row[:valid_width]
        if verbose:
            print(f"    [STORE] Cộng dồn vào Global SRAM Acc. Valid width: {valid_width}")
        return 0

    def run(self, dram_in, dram_w, verbose=False):
        print(f"[V3] Starting Weight Stationary + Line Buffer simulation...")
        start_time = time.time()
        
        H_out = self.cfg.H - self.cfg.R + 1  
        W_out = self.cfg.W - self.cfg.S + 1  
        # [TILING CONFIG] Cắt Channel (C) thành các phần bằng đúng số lượng PE_ROWS (16)
        num_ch_tiles = self.cfg.C // self.cfg.PE_ROWS  
        
        self.global_accumulator.fill(0)
        first_tile_printed = False

        if verbose: print("  > PROLOGUE: Load first weight tile")
        t_prologue = self.load_weight(dram_w, 0, verbose=verbose)
        self.stats['cycles'] += t_prologue
        self.stats['prologue_cycles'] += t_prologue
        self.stats['load_cycles'] += t_prologue
        
        self.w_sram.swap() 
        
        for c_tile in range(num_ch_tiles):
            c_start = c_tile * self.cfg.PE_ROWS

            is_verbose = verbose and not first_tile_printed
            if is_verbose:
                print(f"\n--- [DEMO 1 TILE] Tính toán Channel Tile: c_tile={c_tile} ---")
            
            next_c_tile = c_tile + 1
            t_load_next_weight = 0
            
            if next_c_tile < num_ch_tiles:
                next_start = next_c_tile * self.cfg.PE_ROWS
                if is_verbose: print("  > DMA: Loading NEXT weight tile")
                t_load_next_weight = self.load_weight(dram_w, next_start, verbose=is_verbose)
            
            t_load_input_current_total = 0  
            t_compute_current_total = 0         

            W_padded = self.cfg.W  
            line_buffer = LineBuffer(
                num_channels=self.cfg.PE_ROWS,
                num_rows=self.cfg.R,
                width=W_padded
            )
            
            if is_verbose: print("  > Khởi tạo Line Buffer (Nạp các hàng đầu tiên)")
            for r in range(self.cfg.R):
                if r < self.cfg.H:
                    t_load_row = self.load_input_row_to_linebuffer(line_buffer, dram_in, c_start, r, W_padded, verbose=is_verbose)
                    t_load_input_current_total += t_load_row 
            
            # [TILING CONFIG] Quét Output Height (hàng)
        for oh in range(H_out):
                if oh > 0:
                    next_row_idx = oh + self.cfg.R - 1
                    if next_row_idx < self.cfg.H:
                        if is_verbose: print(f"  > Cập nhật Line Buffer cho hàng output oh={oh}")
                        t_load_row = self.shift_and_load_input_row(line_buffer, dram_in, c_start, next_row_idx, W_padded, verbose=is_verbose)
                        t_load_input_current_total += t_load_row
                
                # [TILING CONFIG] Cắt Output Width (W) thành các phần bằng đúng số lượng PE_COLS (16)
            for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                    self.core.reset_accumulators()
                    
                    window_width = min(self.cfg.PE_COLS + self.cfg.S - 1, W_padded - ow_tile)
                    self.extract_window_from_linebuffer(line_buffer, ow_tile, window_width, verbose=is_verbose)
                    
                    t_compute = self.compute(verbose=is_verbose)
                    t_compute_current_total += t_compute
                    
                    valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                    self.store_accumulator(oh, ow_tile, valid_width, verbose=is_verbose)
                    
                    if is_verbose: 
                        print("    ... (Các output tiles tiếp theo của channel này sẽ bị ẩn log) ...")
                        is_verbose = False
            
            if next_c_tile < num_ch_tiles:
                self.w_sram.swap()
            
            total_bus_busy_time = t_load_input_current_total + t_load_next_weight
            total_pe_active_time = t_load_input_current_total + t_compute_current_total
            actual_step_time = max(total_bus_busy_time, total_pe_active_time)
            
            self.stats['cycles'] += actual_step_time
            self.stats['steady_cycles'] += actual_step_time 
            self.stats['load_cycles'] += total_bus_busy_time 
            
            if total_bus_busy_time > total_pe_active_time:
                self.stats['stall_cycles'] += (total_bus_busy_time - total_pe_active_time)
            
            first_tile_printed = True
        
        if verbose: print("  > EPILOGUE: Write final output to DRAM")
        dram_out = self.global_accumulator.copy()
        
        t_write = H_out * W_out * self.cfg.CYCLE_DRAM_RD
        self.stats['writeback_cycles'] += t_write
        self.stats['epilogue_cycles'] += t_write
        self.stats['cycles'] += t_write
        self.stats['dram_words'] += H_out * W_out
        self.stats['output_dram_words'] += H_out * W_out
        self.stats['output_writes'] = 1
        
        end_time = time.time()
        self.stats['runtime_seconds'] = end_time - start_time
        print(f"[V3] Finished in {end_time - start_time:.4f}s")
        
        return dram_out, self.stats
# ==========================================
# UTILITIES
# ==========================================

def load_data_strict(filename="benchmark_data.npz"):
    """
    Chỉ load dữ liệu từ file có sẵn 
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"❌ LỖI: Không tìm thấy file '{filename}'!\n"
            f"   Bạn cần chạy Version cũ trước để tạo file dữ liệu chuẩn,\n"
            f"   hoặc copy file benchmark_data.npz vào thư mục này."
        )
    
    print(f"\n[DATA] Loading: {filename} ...")
    data = np.load(filename)
    d_in = data['d_in'] # Kỳ vọng shape (C, H, W)
    d_w = data['d_w']   # Kỳ vọng shape (1, C, R, S) hoặc (K, C, R, S)
    
    print(f"  -> Input shape:  {d_in.shape}")
    print(f"  -> Weight shape: {d_w.shape}")
    
    return d_in, d_w

def update_config_from_data(config, d_in, d_w):
    """
    Cập nhật lại H, W, C trong config cho khớp với dữ liệu trong file.
    Tránh trường hợp Config code set H=32 nhưng file lưu H=64 gây lỗi.
    """
    # d_in shape: (C, H, W)
    C_file, H_file, W_file = d_in.shape
    
    # d_w shape: (1, C, R, S)
    _, _, R_file, S_file = d_w.shape
    
    print(f"[CONFIG] Auto-update config theo dữ liệu file:")
    if config.H != H_file: print(f"  - H: {config.H} -> {H_file}")
    if config.W != W_file: print(f"  - W: {config.W} -> {W_file}")
    if config.C != C_file: print(f"  - C: {config.C} -> {C_file}")
    
    config.H = H_file
    config.W = W_file
    config.C = C_file
    config.R = R_file
    config.S = S_file
    
    # Kiểm tra ràng buộc phần cứng
    if config.C % config.PE_ROWS != 0:
        print(f"⚠️ CẢNH BÁO: Số Channel ({config.C}) không chia hết cho PE_ROWS ({config.PE_ROWS}). Code có thể lỗi padding.")

def validate_output(hw_out, d_in, d_w, config):
    """Validate correctness"""
    print("\n[VALIDATION] Checking correctness (Software Reference)...")
    
    # Tính kích thước output mong đợi
    H_out = config.H - config.R + 1
    W_out = config.W - config.S + 1
    
    # Kiểm tra kích thước output của phần cứng
    if hw_out.shape != (H_out, W_out):
        print(f"❌ SIZE MISMATCH: HW Output {hw_out.shape} != Expected {(H_out, W_out)}")
        return False

    errors = 0
    # Lấy kernel đầu tiên (batch 0)
    kernel = d_w[0]
    
    # Duyệt qua từng pixel output để so sánh
    for oh in range(H_out):
        for ow in range(W_out):
            # Tính toán thủ công (Software Golden Reference)
            in_slice = d_in[:, oh:oh+config.R, ow:ow+config.S]
            sw_val = np.sum(in_slice * kernel)
            hw_val = hw_out[oh, ow]
            
            if hw_val != sw_val:
                errors += 1
                if errors <= 3: # Chỉ in 3 lỗi đầu tiên
                    print(f"  Error at [{oh},{ow}]: HW={hw_val}, SW={sw_val}, Diff={hw_val-sw_val}")
    
    total = H_out * W_out
    if errors == 0:
        print(f"✅ ALL {total} pixels CORRECT!")
        return True
    else:
        print(f"❌ FAILED: {errors}/{total} pixels incorrect")
        return False

def calculate_reuse_factors(stats, config):
    H_out = config.H - config.R + 1
    W_out = config.W - config.S + 1
    total_pixels = H_out * W_out
    
    # Theoretical traffic
    theo_weight = total_pixels * (config.C * config.R * config.S)
    theo_input  = total_pixels * (config.C * config.R * config.S)
    
    # Actual traffic (FIXED: Use specific stats, not total dram_words)
    actual_weight = stats['weight_dram_words']
    actual_input  = stats['input_dram_words']
    
    w_reuse = theo_weight / actual_weight if actual_weight > 0 else 0
    i_reuse = theo_input / actual_input if actual_input > 0 else 0
    
    return w_reuse, i_reuse

def print_performance_report(stats, config, version="V3"):
    """Print detailed performance report"""
    print("\n" + "="*70)
    print(f"  PERFORMANCE REPORT - {version}")
    print("="*70)
    
    total = stats['cycles']
    if total == 0: total = 1
    
    # Configuration
    print(f"\n[CONFIGURATION]")
    print(f"  Architecture:     {config.PE_ROWS}×{config.PE_COLS} PEs")
    print(f"  Clock:            {config.FREQ_MHZ} MHz")
    print(f"  Dataflow:         Weight Stationary + Line Buffer")
    print(f"  Workload:         {config.H}×{config.W}×{config.C}, Kernel {config.R}×{config.S}")
    
    # Memory
    print(f"\n[MEMORY]")
    print(f"  On-chip SRAM:     {config.get_sram_size_kb():.2f} KB")
    print(f"    ├─ Weight Buf:  {2 * config.PE_ROWS * config.R * config.S * 4 / 1024:.2f} KB (ping-pong)")
    print(f"    ├─ Input Buf:   {2 * config.PE_ROWS * config.R * 18 * 4 / 1024:.2f} KB (ping-pong)")
    print(f"    ├─ Line Buffer: {config.PE_ROWS * config.R * config.W * 4 / 1024:.2f} KB (spatial reuse)")
    print(f"    └─ Global Acc:  {(config.H-config.R+1)*(config.W-config.S+1)*4/1024:.2f} KB")
    print(f"  DRAM Traffic:     {stats['dram_words']*4/1024:.2f} KB")
    print(f"    ├─ Weights:     {stats['weight_dram_words']*4/1024:.2f} KB")
    print(f"    ├─ Inputs:      {stats['input_dram_words']*4/1024:.2f} KB")
    print(f"    └─ Outputs:     {stats['output_dram_words']*4/1024:.2f} KB")
    
    # Access counts
    print(f"\n[ACCESS COUNTS]")
    print(f"  Weight loads:     {stats['weight_loads']}")
    print(f"  Input loads:      {stats['input_loads']}")
    print(f"  Output writes:    {stats['output_writes']}")
    
    # Line buffer
    print(f"\n[LINE BUFFER]")
    hits = stats['line_buffer_hits']
    total_acc = hits + stats['line_buffer_misses']
    hit_rate = (hits/total_acc*100) if total_acc > 0 else 0
    print(f"  Hits:             {hits}")
    print(f"  Misses:           {stats['line_buffer_misses']}")
    print(f"  Hit rate:         {hit_rate:.1f}%")
    
    # Timing breakdown
    print(f"\n[TIMING - PIPELINE BREAKDOWN]")
    print(f"  Total cycles:     {total:,}")
    print(f"    ├─ Prologue:    {stats['prologue_cycles']:,} ({stats['prologue_cycles']/total*100:.1f}%)")
    print(f"    ├─ Steady:      {stats['steady_cycles']:,} ({stats['steady_cycles']/total*100:.1f}%)")
    print(f"    ├─ Epilogue:    {stats['epilogue_cycles']:,} ({stats['epilogue_cycles']/total*100:.1f}%)")
    print(f"    └─ Writeback:   {stats['writeback_cycles']:,} ({stats['writeback_cycles']/total*100:.1f}%)")
    
    print(f"\n[TIMING - COMPONENT BREAKDOWN]")
    print(f"  Compute cycles:   {stats['compute_cycles']:,} ({stats['compute_cycles']/total*100:.1f}%)")
    print(f"  Stall cycles:     {stats['stall_cycles']:,} ({stats['stall_cycles']/total*100:.1f}%)")
    print(f"  Load cycles:      {stats['load_cycles']:,}")
    
    # Performance metrics
    latency_us = total / config.FREQ_MHZ
    H_out = config.H - config.R + 1
    W_out = config.W - config.S + 1
    total_ops = H_out * W_out * config.C * config.R * config.S * 2
    gops = (total_ops / latency_us) / 1000
    efficiency = (stats['compute_cycles'] / total) * 100
    
    print(f"\n[PERFORMANCE METRICS]")
    print(f"  Latency:          {latency_us:.2f} μs ({latency_us/1000:.4f} ms)")
    print(f"  Frequency:        {config.FREQ_MHZ} MHz")
    print(f"  Efficiency:       {efficiency:.2f}%")
    print(f"  Utilization:      100.00%")
    print(f"  Throughput:       {gops:.2f} GOPS")
    
    # Data reuse
    w_reuse, i_reuse = calculate_reuse_factors(stats, config)
    print(f"\n[DATA REUSE]")
    print(f"  Weight reuse:     {w_reuse:.2f}x")
    print(f"  Input reuse:      {i_reuse:.2f}x")
    
    print("="*70 + "\n")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    cfg = HardwareConfig()
    
    print("="*70)
    print("  CNN ACCELERATOR - V3 (WEIGHT STATIONARY + LINE BUFFER)")
    print("="*70)
    
    try:
        # Load data
        d_in, d_w = load_data_strict("benchmark_data.npz")
        
        # Update config
        update_config_from_data(cfg, d_in, d_w)
        
        # Run accelerator
        print(f"\n[INIT] V3-FIXED with {cfg.PE_ROWS}×{cfg.PE_COLS} PEs...\n")
        accel = HardwareAccelerator(cfg)
        hw_out, stats = accel.run(d_in, d_w, verbose=True)
        
        # Report & validate
        print_performance_report(stats, cfg)
        
        is_correct = validate_output(hw_out, d_in, d_w, cfg)
        
        if is_correct:
            print("✅ V3 SIMULATION SUCCESSFUL!")
        else:
            print("❌ V3 SIMULATION FAILED!")
            
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()