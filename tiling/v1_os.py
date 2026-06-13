import numpy as np
import time
from dataclasses import dataclass
import os

# ==========================================
# VERSION 1: OUTPUT STATIONARY (BASELINE)
# ==========================================

# ==========================================
# 1. HARDWARE CONFIGURATION 
# ==========================================
@dataclass
class HardwareConfig:
    """Cấu hình phần cứng cho CNN Accelerator"""
    # Kích thước bài toán
    H: int = 32
    W: int = 32
    C: int = 128
    R: int = 3
    S: int = 3
    
    # Kiến trúc phần cứng
    PE_ROWS: int = 16   # Số hàng PE (unroll input channels)
    PE_COLS: int = 16   # Số cột PE (unroll output width)
    DATA_WIDTH: int = 32  # 32-bit integer
    
    # Timing parameters
    FREQ_MHZ: float = 200.0
    CYCLE_MAC: int = 1        # Pipelined MAC unit
    CYCLE_DRAM_RD: int = 1    # DRAM read or write per word
    
    def get_sram_size_kb(self):
        """Tính kích thước SRAM on-chip (double buffered)"""
        # Double Buffer (x2)
        # Weight Buffer: Tile Channels x R x S = 16x3x3
        w_bits = 2 * (self.PE_ROWS * self.R * self.S) * self.DATA_WIDTH
        
        # Input buffer: 16 channels × 3 × 18 (width + halo) × 2 banks
        in_req_w = self.PE_COLS + self.S - 1
        in_bits = 2 * (self.PE_ROWS * self.R * in_req_w) * self.DATA_WIDTH
        
        return (w_bits + in_bits) / 8192  # Convert to KB

# ==========================================
# 2.MEMORY SYSTEM
# ==========================================
class PingPongSRAM:
    """
    Double-buffered SRAM với ping-pong mechanism.
    Cho phép load data mới trong khi PE đang compute với data cũ.
    """
    def __init__(self, shape, name="SRAM"):
        self.name = name
        # Hai banks: Bank 0 và Bank 1
        self.banks = [
            np.zeros(shape, dtype=np.int32), 
            np.zeros(shape, dtype=np.int32)
        ]
        #MUX chọn kênh
        self.compute_idx = 0  # Bank đang được PE đọc
        self.load_idx = 1     # Bank đang được DMA ghi

    def swap(self):
        """Swap banks: compute ↔ load"""
        self.compute_idx = 1 - self.compute_idx
        self.load_idx = 1 - self.load_idx

    def write_from_dram(self, data):
        """DMA ghi dữ liệu từ DRAM vào load bank"""
        np.copyto(self.banks[self.load_idx], data)

    def read_to_pe(self):
        """PE đọc dữ liệu từ compute bank"""
        return self.banks[self.compute_idx]

# ==========================================
# 3. COMPUTE CORE (PE ARRAY & ADDER TREE)
# ==========================================
class ComputeCore:
    """
    PE Array 16x16 với MAC units và Adder Tree.
    Thực hiện convolution với weight broadcasting và input sliding window.
    """
    def __init__(self, config):
        self.cfg = config
        # Accumulator registers trong mỗi PE
        self.accumulators = np.zeros(
            (config.PE_ROWS, config.PE_COLS), 
            dtype=np.int32
        )

    def reset_accumulators(self):
        """Reset tất cả accumulators về 0"""
        self.accumulators.fill(0)

    def execute_cycle_accurate(self, weight_tile, input_tile):
        """
        Thực hiện MAC operations với:
        - weight_tile: shape (16, 3, 3)
        - input_tile: shape (16, 3, 18)
        """
        # Loop qua kernel 3×3
        for r in range(self.cfg.R):
            for s in range(self.cfg.S):
                # Weight broadcasting: (16,) → (16, 1) → broadcast to (16, 16)
                w_vec = weight_tile[:, r, s].reshape(self.cfg.PE_ROWS, 1)
                
                # Hardware MUX/Shifter logic:
                # Input sliding window: lấy 16 cột liên tiếp
                start_col = s
                end_col = s + self.cfg.PE_COLS
                in_mat = input_tile[:, r, start_col:end_col]
                
                # MAC: Accumulator += Weight × Input
                self.accumulators += w_vec * in_mat

    def adder_tree_reduction_structural(self):
        """
        Cây cộng 4 tầng để reduction 16 channels → 1 output row.
        Mô phỏng cấu trúc phần cứng thực tế.
        """
        # Tầng 0: 16 inputs
        layer_0 = [self.accumulators[r, :] for r in range(16)]
        
        # Tầng 1: 16 → 8 (8 adders song song)
        layer_1 = []
        for i in range(0, 16, 2):
            layer_1.append(layer_0[i] + layer_0[i+1])
        
        # Tầng 2: 8 → 4 (4 adders song song)
        layer_2 = []
        for i in range(0, 8, 2):
            layer_2.append(layer_1[i] + layer_1[i+1])
        
        # Tầng 3: 4 → 2 (2 adders song song)
        layer_3 = []
        for i in range(0, 4, 2):
            layer_3.append(layer_2[i] + layer_2[i+1])
        
        # Tầng 4: 2 → 1 (1 adder)
        final_result = layer_3[0] + layer_3[1]
        
        return final_result

# ==========================================
# 4. TOP LEVEL CONTROLLER (ACCELERATOR)
# ==========================================
class HardwareAccelerator:
    """
    Top-level CNN accelerator controller.
    Dataflow: Output Stationary (OS)
    """
    def __init__(self, config):
        self.cfg = config
        
        # Memory hierarchy
        self.w_sram = PingPongSRAM(
            (config.PE_ROWS, config.R, config.S), 
            "WeightBuf"
        )
        
        input_buf_w = config.PE_COLS + config.S - 1  # 16 + 2 = 18
        self.in_sram = PingPongSRAM(
            (config.PE_ROWS, config.R, input_buf_w), 
            "InputBuf"
        )
        
        # Compute unit
        self.core = ComputeCore(config)
        
        # Performance metrics
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

    def load_input(self, dram_in, c_start, h_start, h_end, width_start, width_end, verbose=False):
        in_slice_padded = np.zeros(
            (16, self.cfg.R, self.cfg.PE_COLS + self.cfg.S - 1), 
            dtype=np.int32
        )
        # [TILING CONFIG] Lấy 1 khối Input Tile cắt theo C (16 kênh), Height và Width
        actual_data = dram_in[c_start:c_start+16, h_start:h_end, width_start:width_end]
        in_slice_padded[:, :actual_data.shape[1], :actual_data.shape[2]] = actual_data
        
        self.in_sram.write_from_dram(in_slice_padded)
        
        t_load = in_slice_padded.size * self.cfg.CYCLE_DRAM_RD
        self.stats['dram_words'] += in_slice_padded.size
        self.stats['input_dram_words'] += in_slice_padded.size
        self.stats['input_loads'] += 1
        
        if verbose:
            print(f"    [LOAD INPUT] Vị trí H:{h_start}-{h_end}, W:{width_start}-{width_end}. Shape: {in_slice_padded.shape}, Words: {in_slice_padded.size}, Bus Cycles: {t_load}")
        return t_load

    def compute(self, verbose=False):
        curr_w = self.w_sram.read_to_pe()
        curr_in = self.in_sram.read_to_pe()
        
        t_compute = self.cfg.R * self.cfg.S * self.cfg.CYCLE_MAC
        self.stats['compute_cycles'] += t_compute
        
        self.core.execute_cycle_accurate(curr_w, curr_in)
        if verbose:
            print(f"    [COMPUTE] MAC window {self.cfg.R}x{self.cfg.S}. PE Cycles: {t_compute}")
        return t_compute

    def store(self, dram_out, oh, ow_tile, valid_width, verbose=False):
        output_row_vals = self.core.adder_tree_reduction_structural()
        dram_out[oh, ow_tile:ow_tile + valid_width] = output_row_vals[:valid_width]
        
        t_write = valid_width * self.cfg.CYCLE_DRAM_RD
        self.stats['dram_words'] += valid_width
        self.stats['output_dram_words'] += valid_width
        self.stats['output_writes'] += 1
        
        if verbose:
            print(f"    [STORE] Ghi kết quả ra DRAM. Valid width: {valid_width}, Words: {valid_width}, Bus Cycles: {t_write}")
        return t_write

    def run(self, dram_in, dram_w, verbose=False):
        print(f"[V1] Starting Output Stationary simulation...")
        start_time = time.time()
        
        H_out = self.cfg.H - self.cfg.R + 1  
        W_out = self.cfg.W - self.cfg.S + 1  
        dram_out = np.zeros((H_out, W_out), dtype=np.int32)
        # [TILING CONFIG] Cắt Channel (C) thành các phần bằng đúng số lượng PE_ROWS (16)
        num_ch_tiles = self.cfg.C // self.cfg.PE_ROWS  
        
        first_tile_printed = False

        # [TILING CONFIG] Quét Output Height (hàng)
        for oh in range(H_out):
            # [TILING CONFIG] Cắt Output Width (W) thành các phần bằng đúng số lượng PE_COLS (16)
            for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                
                self.core.reset_accumulators()
                
                is_verbose = verbose and not first_tile_printed
                if is_verbose:
                    print(f"\n--- [DEMO 1 TILE] Tính toán Output Tile: oh={oh}, ow_tile={ow_tile} ---")
                
                if is_verbose: print("  > STAGE 1: PROLOGUE (DMA nạp dữ liệu khởi đầu)")
                t_load_w = self.load_weight(dram_w, 0, verbose=is_verbose)
                
                width_end = min(ow_tile + self.cfg.PE_COLS + self.cfg.S - 1, self.cfg.W)
                t_load_in = self.load_input(dram_in, 0, oh, oh + self.cfg.R, ow_tile, width_end, verbose=is_verbose)
                
                t_prologue = t_load_w + t_load_in
                self.stats['prologue_cycles'] += t_prologue
                self.stats['cycles'] += t_prologue
                self.stats['load_cycles'] += t_prologue
                
                self.w_sram.swap()
                self.in_sram.swap()
                
                if is_verbose: print("  > STAGE 2: STEADY STATE (Pipeline Compute và Load song song)")
                for k in range(num_ch_tiles - 1):
                    t_compute = self.compute(verbose=is_verbose)
                    
                    next_k = k + 1
                    c_start = next_k * self.cfg.PE_ROWS
                    
                    if is_verbose: print(f"    (Song song nạp tile tiếp theo k={next_k})")
                    t_load_w = self.load_weight(dram_w, c_start, verbose=is_verbose)
                    t_load_in = self.load_input(dram_in, c_start, oh, oh + self.cfg.R, ow_tile, width_end, verbose=is_verbose)
                    t_load = t_load_w + t_load_in
                    
                    t_steady_step = max(t_compute, t_load)
                    t_stall = max(0, t_load - t_compute)
                    
                    self.stats['steady_cycles'] += t_steady_step     
                    self.stats['cycles'] += t_steady_step
                    self.stats['load_cycles'] += t_load             
                    self.stats['stall_cycles'] += t_stall
                    
                    self.w_sram.swap()
                    self.in_sram.swap()
                
                if is_verbose: print("  > STAGE 3: EPILOGUE (Compute tile cuối)")
                t_epilogue = self.compute(verbose=is_verbose)
                self.stats['epilogue_cycles'] += t_epilogue  
                self.stats['cycles'] += t_epilogue
                
                if is_verbose: print("  > STAGE 4: WRITE BACK (Ghi kết quả)")
                valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                t_write = self.store(dram_out, oh, ow_tile, valid_width, verbose=is_verbose)
                
                self.stats['writeback_cycles'] += t_write  
                self.stats['cycles'] += t_write
                
                first_tile_printed = True
        
        end_time = time.time()
        self.stats['runtime_seconds'] = end_time - start_time
        print(f"[V1] Finished in {end_time - start_time:.4f}s")
        
        return dram_out, self.stats
# ==========================================
# UTILITIES
# ==========================================

def load_or_generate_data(config, filename="benchmark_data.npz"):
    """
    Load hoặc generate test data.
    Đảm bảo consistency giữa các versions.
    """
    if os.path.exists(filename):
        print(f"[DATA] Loading existing data: {filename}")
        data = np.load(filename)
        
        # Validate shape
        if data['d_in'].shape != (config.C, config.H, config.W):
            print(f"[WARNING] Shape mismatch! Regenerating...")
        else:
            return data['d_in'], data['d_w']
    
    print(f"[DATA] Generating new random data...")
    
    # Simple random data (dễ debug)
    d_in = np.random.randint(0, 5, (config.C, config.H, config.W), dtype=np.int32)
    d_w = np.random.randint(0, 5, (1, config.C, config.R, config.S), dtype=np.int32)
    
    # Save for future use
    np.savez(filename, d_in=d_in, d_w=d_w)
    print(f"[DATA] Saved to {filename}")
    
    return d_in, d_w

def validate_output(hw_out, d_in, d_w, config):
    """
    Validate toàn bộ output 
    """
    print("\n[VALIDATION] Checking all output pixels...")
    
    H_out = config.H - config.R + 1
    W_out = config.W - config.S + 1
    
    errors = 0
    max_error = 0
    
    for oh in range(H_out):
        for ow in range(W_out):
            # Software reference
            h_start = oh
            w_start = ow
            sw_val = np.sum(
                d_in[:, h_start:h_start+config.R, w_start:w_start+config.S] * d_w[0]
            )
            
            hw_val = hw_out[oh, ow]
            
            if hw_val != sw_val:
                errors += 1
                diff = abs(hw_val - sw_val)
                max_error = max(max_error, diff)
                
                # Print first few errors
                if errors <= 3:
                    print(f"  Error at [{oh},{ow}]: HW={hw_val}, SW={sw_val}, Diff={diff}")
    
    total_pixels = H_out * W_out
    
    if errors == 0:
        print(f"✅ ALL {total_pixels} pixels CORRECT!")
        return True
    else:
        print(f"❌ FAILED: {errors}/{total_pixels} pixels incorrect")
        print(f"   Max error: {max_error}")
        return False

def calculate_reuse_factors(stats, config):
    """Tính toán data reuse factors"""
    H_out = config.H - config.R + 1
    W_out = config.W - config.S + 1
    total_output_pixels = H_out * W_out
    
    # Theoretical usage (nếu mỗi pixel load riêng)
    theoretical_weight_usage = total_output_pixels * config.C
    theoretical_input_usage = total_output_pixels * config.C
    
    # Actual loads
    actual_weight_loads = stats['weight_loads']
    actual_input_loads = stats['input_loads']
    
    # Reuse factors
    weight_reuse = (theoretical_weight_usage / (actual_weight_loads * config.PE_ROWS) 
                    if actual_weight_loads > 0 else 0)
    input_reuse = (theoretical_input_usage / (actual_input_loads * config.PE_ROWS) 
                   if actual_input_loads > 0 else 0)
    
    return {
        'weight_reuse_factor': weight_reuse,
        'input_reuse_factor': input_reuse,
    }


def print_performance_report(stats, config):
    """
    Enhanced performance report với breakdown chi tiết.
    """
    print("\n" + "="*70)
    print("  PERFORMANCE REPORT - OUTPUT STATIONARY")
    print("="*70)
    
    # Basic info
    print(f"\n[CONFIGURATION]")
    print(f"  Architecture:     {config.PE_ROWS}×{config.PE_COLS} PEs (Systolic Array)")
    print(f"  Clock:            {config.FREQ_MHZ} MHz")
    print(f"  Dataflow:         Output Stationary")
    print(f"  Workload:         {config.H}×{config.W}×{config.C}, Kernel {config.R}×{config.S}")
    
    # Memory analysis
    print(f"\n[MEMORY]")
    print(f"  On-chip SRAM:     {config.get_sram_size_kb():.2f} KB")
    print(f"    ├─ Weight Buf:  {2 * config.PE_ROWS * config.R * config.S * 4 / 1024:.2f} KB")
    print(f"    └─ Input Buf:   {2 * config.PE_ROWS * config.R * 18 * 4 / 1024:.2f} KB")
    
    print(f"  DRAM Traffic:     {stats['dram_words'] * 4 / 1024:.2f} KB")
    print(f"    ├─ Weights:     {stats['weight_dram_words'] * 4 / 1024:.2f} KB")
    print(f"    ├─ Inputs:      {stats['input_dram_words'] * 4 / 1024:.2f} KB")
    print(f"    └─ Outputs:     {stats['output_dram_words'] * 4 / 1024:.2f} KB")
    
    # Access counts
    print(f"\n[ACCESS COUNTS]")
    print(f"  Weight loads:     {stats['weight_loads']}")
    print(f"  Input loads:      {stats['input_loads']}")
    print(f"  Output writes:    {stats['output_writes']}")
    
    # Timing : Pipeline Breakdown
    total = stats['cycles']
    prologue = stats['prologue_cycles']
    steady = stats['steady_cycles']
    epilogue = stats['epilogue_cycles']
    writeback = stats['writeback_cycles']
    
    print(f"\n[TIMING - PIPELINE BREAKDOWN]")
    print(f"  Total cycles:     {total:,}")
    print(f"    ├─ Prologue:    {prologue:,} ({prologue/total*100:.1f}%) - Load only")
    print(f"    ├─ Steady:      {steady:,} ({steady/total*100:.1f}%) - Overlapped")
    print(f"    ├─ Epilogue:    {epilogue:,} ({epilogue/total*100:.1f}%) - Compute only")
    print(f"    └─ Writeback:   {writeback:,} ({writeback/total*100:.1f}%) - Output")
    
    compute = stats['compute_cycles']
    stall = stats['stall_cycles']

    print(f"\n[TIMING - COMPONENT BREAKDOWN]")
    print(f"  Compute cycles:   {compute:,} ({compute/total*100:.1f}%)")
    print(f"  Stall cycles:     {stall:,} ({stall/total*100:.1f}%)")
    print(f"  Load cycles:      {stats['load_cycles']:,} (overlaps with compute)")

    print(f"\n[PERFORMANCE METRICS]")
    latency_us = total / config.FREQ_MHZ
    latency_ms = latency_us / 1000
    print(f"  Latency:          {latency_us:.2f} μs ({latency_ms:.4f} ms)")
    print(f"  Frequency:        {config.FREQ_MHZ} MHz")
    print(f"  Efficiency:       {compute/total*100:.2f}% (compute/total)")
    print(f"  Utilization:      {(total - stall)/total*100:.2f}% (non-stall/total)")
    # Throughput
    H_out = config.H - config.R + 1
    W_out = config.W - config.S + 1
    total_ops = H_out * W_out * config.C * config.R * config.S
    throughput_gops = (total_ops / latency_us) / 1000  # GOPS
    
    print(f"  Throughput:       {throughput_gops:.2f} GOPS")
    
    # Reuse analysis
    reuse = calculate_reuse_factors(stats, config)
    print(f"\n[DATA REUSE]")
    print(f"  Weight reuse:     {reuse['weight_reuse_factor']:.2f}x")
    print(f"  Input reuse:      {reuse['input_reuse_factor']:.2f}x")
    
    print("="*70 + "\n")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # Configuration
    cfg = HardwareConfig()
    
    print("="*70)
    print("  CNN ACCELERATOR SIMULATOR ")
    print("="*70)
    
    # Load/generate data
    d_in, d_w = load_or_generate_data(cfg, filename="benchmark_data.npz")
    
    # Initialize accelerator
    print(f"\n[INIT] Initializing {cfg.PE_ROWS}×{cfg.PE_COLS} PE array...")
    accel = HardwareAccelerator(cfg)
    
    # Run simulation
    hw_out, stats = accel.run(d_in, d_w, verbose=True)
    
    # Print report
    print_performance_report(stats, cfg)
    
    # Validate correctness
    is_correct = validate_output(hw_out, d_in, d_w, cfg)
    
    if is_correct:
        print("✅ SIMULATION SUCCESSFUL - All outputs correct!")
    else:
        print("❌ SIMULATION FAILED - Output mismatch detected!")