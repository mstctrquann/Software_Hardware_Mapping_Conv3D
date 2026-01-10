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
        
        # Performance metrics (IMPROVED - More detailed!)
        self.stats = {
            # Timing
            "cycles": 0,
            # Breakdown by pipeline stage
            "prologue_cycles": 0,     # Load only
            "steady_cycles": 0,       # Overlapped load+compute
            "epilogue_cycles": 0,     # Compute only
            "writeback_cycles": 0,    # Output writes

            # Component cycles
            "compute_cycles": 0,      # Total compute time
            "load_cycles": 0,         # Total load time
            "stall_cycles": 0,        # PE idle time = load - compute
            
            # DRAM traffic (total)
            "dram_words": 0,
            
            # DRAM traffic breakdown by type
            "weight_dram_words": 0,
            "input_dram_words": 0,
            "output_dram_words": 0,
            
            # Access counts => số lần ghi, đọc từ DRAM
            "weight_loads": 0,
            "input_loads": 0,
            "output_writes": 0,
        }

    def run(self, dram_in, dram_w):
        """
        Main execution loop.
        Dataflow: Output Stationary với 3-stage pipeline.
        """
        print(f"[V1] Starting Output Stationary simulation...")
        start_time = time.time()
        
        # Derived dimensions
        H_out = self.cfg.H - self.cfg.R + 1  # 30
        W_out = self.cfg.W - self.cfg.S + 1  # 30
        dram_out = np.zeros((H_out, W_out), dtype=np.int32)
        
        num_ch_tiles = self.cfg.C // self.cfg.PE_ROWS  # 8 tiles
        
        # ==========================================
        # DATAFLOW: Output Stationary
        # Loop order: Output Height → Output Width → Channels
        # ==========================================
        
        for oh in range(H_out):
            for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                
                # Reset accumulators cho output tile mới
                self.core.reset_accumulators()
                
                # ==========================================
                # STAGE 1: PROLOGUE - Load first tile
                # ==========================================
                
                # Load weight tile 0
                w_slice = dram_w[0, 0:16, :, :]
                
                # Load input tile 0 với zero padding
                h_start, h_end = oh, oh + self.cfg.R
                width_start = ow_tile
                width_end = min(
                    ow_tile + self.cfg.PE_COLS + self.cfg.S - 1, 
                    self.cfg.W
                )
                
                # Khởi tạo in_slice toàn số 0 với kích thước chuẩn (16, 3, 18) để tránh trường hợp không đủ 18ptu ở rìa
                in_slice_padded = np.zeros(
                    (16, self.cfg.R, self.cfg.PE_COLS + self.cfg.S - 1), 
                    dtype=np.int32
                )
                # Lấy dữ liệu thực tế từ DRAM (có thể chỉ được 16 hoặc ít hơn)
                actual_data = dram_in[0:16, h_start:h_end, width_start:width_end]

                # Chép dữ liệu thực tế vào đầu mảng in_slice_padded
                # Những phần thiếu ở cuối sẽ giữ nguyên là 0 (đúng Zero Padding)
                in_slice_padded[:, :actual_data.shape[1], :actual_data.shape[2]] = actual_data
                
                # DMA write to SRAM
                self.w_sram.write_from_dram(w_slice)
                self.in_sram.write_from_dram(in_slice_padded)
                
                # Tính Cycle Load ban đầu (Stall hoàn toàn)
                # Do chỉ nạp data từ DRAM và các PE k tính toán => latency được cộng thẳng
                t_prologue = (w_slice.size + in_slice_padded.size) * self.cfg.CYCLE_DRAM_RD
                self.stats['prologue_cycles'] += t_prologue
                self.stats['cycles'] += t_prologue
                self.stats['load_cycles'] += t_prologue

                self.stats['dram_words'] += (w_slice.size + in_slice_padded.size)
                
                # Track breakdown
                self.stats['weight_dram_words'] += w_slice.size
                self.stats['input_dram_words'] += in_slice_padded.size
                self.stats['weight_loads'] += 1
                self.stats['input_loads'] += 1
                
                # Hoàn thành nạp dữ liệu cho tile 0 => Swap để đưa Tile 0 vào Compute Bank để bắt đầu tính toán
                self.w_sram.swap()
                self.in_sram.swap()
                
                # ==========================================
                # STAGE 2: STEADY STATE - Pipeline compute/load
                # ==========================================
                
                for k in range(num_ch_tiles - 1):
                    # A. COMPUTE với tile hiện tại
                    curr_w = self.w_sram.read_to_pe()
                    curr_in = self.in_sram.read_to_pe()
                    
                    # Compute time
                    t_compute = self.cfg.R * self.cfg.S * self.cfg.CYCLE_MAC
                    
                    # B. LOAD tile tiếp theo (song song với compute)
                    next_k = k + 1
                    c_start = next_k * self.cfg.PE_ROWS
                    
                    # Load next weight tile
                    w_next = dram_w[0, c_start:c_start+16, :, :]
                    
                    # Load next input tile
                    in_next_padded = np.zeros(
                        (16, self.cfg.R, self.cfg.PE_COLS + self.cfg.S - 1), 
                        dtype=np.int32
                    )
                    actual_data_in = dram_in[
                        c_start:c_start+16, 
                        h_start:h_end, 
                        width_start:width_end
                    ]
                    in_next_padded[:, :actual_data_in.shape[1], :actual_data_in.shape[2]] = actual_data_in
                    
                    # DMA write
                    self.w_sram.write_from_dram(w_next)
                    self.in_sram.write_from_dram(in_next_padded)
                    
                    # Load time
                    t_load = (w_next.size + in_next_padded.size) * self.cfg.CYCLE_DRAM_RD
                    
                    # Track metrics
                    self.stats['dram_words'] += (w_next.size + in_next_padded.size)
                    self.stats['weight_dram_words'] += w_next.size
                    self.stats['input_dram_words'] += in_next_padded.size
                    self.stats['weight_loads'] += 1
                    self.stats['input_loads'] += 1
                    
                    # C. LATENCY HIDING - Pipeline barrier
                    t_steady_step = max(t_compute, t_load)
                    t_stall = max(0, t_load - t_compute)
                    
                    self.stats['steady_cycles'] += t_steady_step     
                    self.stats['cycles'] += t_steady_step
                    self.stats['compute_cycles'] += t_compute
                    self.stats['load_cycles'] += t_load             
                    self.stats['stall_cycles'] += t_stall
                    
                    # Perform computation (functional simulation)
                    self.core.execute_cycle_accurate(curr_w, curr_in)
                    
                    # Swap banks
                    self.w_sram.swap()
                    self.in_sram.swap()
                
                # ==========================================
                # STAGE 3: EPILOGUE - Compute last tile
                # ==========================================
                
                curr_w = self.w_sram.read_to_pe()
                curr_in = self.in_sram.read_to_pe()
                
                # Epilogue
                t_epilogue = self.cfg.R * self.cfg.S * self.cfg.CYCLE_MAC

                self.stats['epilogue_cycles'] += t_epilogue  
                self.stats['cycles'] += t_epilogue
                self.stats['compute_cycles'] += t_epilogue

                self.core.execute_cycle_accurate(curr_w, curr_in)
                
                # ==========================================
                # WRITE BACK - Reduction và ghi DRAM
                # ==========================================
                
                # Adder tree reduction: 16 channels → 1 output row
                output_row_vals = self.core.adder_tree_reduction_structural()
                
                # Ghi ra DRAM
                valid_width = min(self.cfg.PE_COLS, W_out - ow_tile) #W_out - ow_tile:số ptu còn lại nếu k đủ 16ptu để ghi
                dram_out[oh, ow_tile:ow_tile + valid_width] = output_row_vals[:valid_width]

                # Writeback
                t_write = valid_width * self.cfg.CYCLE_DRAM_RD

                self.stats['writeback_cycles'] += t_write  
                self.stats['cycles'] += t_write
                self.stats['dram_words'] += valid_width
                self.stats['output_dram_words'] += valid_width
                self.stats['output_writes'] += 1
        
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
    hw_out, stats = accel.run(d_in, d_w)
    
    # Print report
    print_performance_report(stats, cfg)
    
    # Validate correctness
    is_correct = validate_output(hw_out, d_in, d_w, cfg)
    
    if is_correct:
        print("✅ SIMULATION SUCCESSFUL - All outputs correct!")
    else:
        print("❌ SIMULATION FAILED - Output mismatch detected!")