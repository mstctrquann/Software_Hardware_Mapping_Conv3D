import numpy as np
import time
from dataclasses import dataclass
import os

# ==========================================
# VERSION 2: WEIGHT STATIONARY
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
        
        # Global accumulator buffer 
        H_out = self.H - self.R + 1
        W_out = self.W - self.S + 1
        acc_bits = H_out * W_out * self.DATA_WIDTH
        
        return (w_bits + in_bits + acc_bits) / 8192

# ==========================================
# 2. MEMORY SYSTEM
# ==========================================
class PingPongSRAM:
    """
    Double-buffered SRAM với ping-pong mechanism.
    """
    def __init__(self, shape, name="SRAM"):
        self.name = name
        self.banks = [
            np.zeros(shape, dtype=np.int32), 
            np.zeros(shape, dtype=np.int32)
        ]
        self.compute_idx = 0
        self.load_idx = 1

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
# 3. COMPUTE CORE
# ==========================================
class ComputeCore:
    """
    PE Array 16×16 với MAC units và Adder Tree.
    """
    def __init__(self, config):
        self.cfg = config
        self.accumulators = np.zeros(
            (config.PE_ROWS, config.PE_COLS), 
            dtype=np.int32
        )

    def reset_accumulators(self):
        """Reset tất cả accumulators về 0"""
        self.accumulators.fill(0)

    def execute_cycle_accurate(self, weight_tile, input_tile):
        """
        Thực hiện MAC operations
        """
        # weight_tile: 16x3x3, input_tile:16x3x18
        for r in range(self.cfg.R):
            for s in range(self.cfg.S):
                w_vec = weight_tile[:, r, s].reshape(self.cfg.PE_ROWS, 1) #w_vec: 16x1
                start_col = s
                end_col = s + self.cfg.PE_COLS
                in_mat = input_tile[:, r, start_col:end_col] #in_mat= 16x1x16
                self.accumulators += w_vec * in_mat

    def adder_tree_reduction_structural(self):
        """
        Cây cộng 4 tầng: 16 rows → 1 output row
        """
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
# 4. TOP LEVEL CONTROLLER -  WEIGHT STATIONARY
# ==========================================
class HardwareAccelerator:
    """
    Top-level CNN accelerator controller.
    Dataflow: Weight Stationary (WS)
    
    Key difference from V1:
    - Loop order: Channel → Height → Width (instead of H→W→C)
    - Weights loaded ONCE per channel tile, reused for all output pixels
    """
    def __init__(self, config):
        self.cfg = config
        
        # Memory hierarchy (same as V1)
        self.w_sram = PingPongSRAM(
            (config.PE_ROWS, config.R, config.S), 
            "WeightBuf"
        )
        
        input_buf_w = config.PE_COLS + config.S - 1
        self.in_sram = PingPongSRAM(
            (config.PE_ROWS, config.R, input_buf_w), 
            "InputBuf"
        )
        
        # Compute unit
        self.core = ComputeCore(config)

        # Global output accumulator (on-chip SRAM)
        H_out = config.H - config.R + 1
        W_out = config.W - config.S + 1
        self.global_accumulator = np.zeros((H_out, W_out), dtype=np.int32)
        
        # Performance metrics
        self.stats = {
            # Total cycles
            "cycles": 0,
            
            # Pipeline stages
            "prologue_cycles": 0,
            "steady_cycles": 0,
            "epilogue_cycles": 0,
            "writeback_cycles": 0,
            
            # Component cycles
            "compute_cycles": 0,
            "load_cycles": 0,
            "stall_cycles": 0,
            
            # DRAM traffic
            "dram_words": 0,
            "weight_dram_words": 0,
            "input_dram_words": 0,
            "output_dram_words": 0,
            
            # Access counts
            "weight_loads": 0,
            "input_loads": 0,
            "output_writes": 0,
        }

    def run(self, dram_in, dram_w):
        """
        Main execution loop - Weight Stationary dataflow
        
        Loop order: Channel tiles → Output Height → Output Width
        """
        print(f"[V2] Starting Weight Stationary simulation...")
        start_time = time.time()
        
        # Derived dimensions
        H_out = self.cfg.H - self.cfg.R + 1  # 30
        W_out = self.cfg.W - self.cfg.S + 1  # 30
        num_ch_tiles = self.cfg.C // self.cfg.PE_ROWS  # 8 tiles
        
        # Reset global accumulator 
        self.global_accumulator.fill(0)

        # ==========================================
        # PROLOGUE: Load first weight tile
        # ==========================================
        
        print("Prologue: Loading first weight tile...")
        first_w_slice = dram_w[0, 0:16, :, :]
        self.w_sram.write_from_dram(first_w_slice)
        
        t_prologue = first_w_slice.size * self.cfg.CYCLE_DRAM_RD
        self.stats['cycles'] += t_prologue
        self.stats['prologue_cycles'] += t_prologue
        self.stats['load_cycles'] += t_prologue
        
        self.stats['dram_words'] += first_w_slice.size
        self.stats['weight_dram_words'] += first_w_slice.size
        self.stats['weight_loads'] += 1
        
        # Swap: weight[0] now in compute bank
        self.w_sram.swap()
        
        # ==========================================
        # OUTER LOOP: CHANNEL TILES (Weight Stationary)
        # ==========================================
        
        for c_tile in range(num_ch_tiles):
            c_start = c_tile * self.cfg.PE_ROWS
            
            # -----------------------------------------------------------
            # Load NEXT weight tile (DMA, overlap with compute)
            # -----------------------------------------------------------
            next_c_tile = c_tile + 1
            t_load_next_weight = 0
            
            if next_c_tile < num_ch_tiles:
                next_start = next_c_tile * self.cfg.PE_ROWS
                w_next_slice = dram_w[0, next_start:next_start+16, :, :]
                
                # Write to load bank
                self.w_sram.write_from_dram(w_next_slice)
                
                t_load_next_weight = w_next_slice.size * self.cfg.CYCLE_DRAM_RD
                
                self.stats['dram_words'] += w_next_slice.size
                self.stats['weight_dram_words'] += w_next_slice.size
                self.stats['weight_loads'] += 1
                
                print(f"  ->Loading weight[{next_c_tile}] to load bank (DMA)")

                t_load_input_total = 0
                t_compute_total = 0
            
            # ==========================================
            # INNER LOOPS: ALL OUTPUT PIXELS
            # Reuse same weights
            # ==========================================
            
            for oh in range(H_out):
                for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                    
                    # Reset accumulators for new output tile
                    self.core.reset_accumulators()
                    
                    # Spatial window for input
                    h_start, h_end = oh, oh + self.cfg.R
                    width_start = ow_tile
                    width_end = min(
                        ow_tile + self.cfg.PE_COLS + self.cfg.S - 1, 
                        self.cfg.W
                    )
                    
                    # Load input from DRAM
                    in_slice_padded = np.zeros(
                        (16, self.cfg.R, self.cfg.PE_COLS + self.cfg.S - 1),
                        dtype=np.int32
                    )
                    
                    actual_data = dram_in[
                        c_start:c_start+16,
                        h_start:h_end,
                        width_start:width_end
                    ]
                    
                    in_slice_padded[:, :actual_data.shape[1], :actual_data.shape[2]] = actual_data
                    
                    # Write to input SRAM
                    self.in_sram.write_from_dram(in_slice_padded)
                    self.in_sram.swap()
                    
                    # Track input load
                    t_load_input = in_slice_padded.size * self.cfg.CYCLE_DRAM_RD
                    t_load_input_total += t_load_input
                    
                    self.stats['dram_words'] += in_slice_padded.size
                    self.stats['input_dram_words'] += in_slice_padded.size
                    self.stats['input_loads'] += 1
                    
                    # Compute
                    curr_w = self.w_sram.read_to_pe()
                    curr_in = self.in_sram.read_to_pe()
                    
                    t_compute = self.cfg.R * self.cfg.S * self.cfg.CYCLE_MAC
                    t_compute_total += t_compute
                    self.stats['compute_cycles'] += t_compute
                    
                    self.core.execute_cycle_accurate(curr_w, curr_in)
                    
                    # ==========================================
                    # ACCUMULATE - Add to global buffer 
                    # ==========================================
                    
                    # Adder tree reduction
                    output_row_vals = self.core.adder_tree_reduction_structural()
                    
                    # Accumulate in global buffer (on-chip SRAM)
                    valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                    self.global_accumulator[oh, ow_tile:ow_tile + valid_width] += output_row_vals[:valid_width]
            # -----------------------------------------------------------
            # Calculate cycles for this tile
            # -----------------------------------------------------------
            
            # Timeline:
            # |---Load Input + Compute---|
            #                            |---Load Next Weight---|
            #
            # PE time: Load Input + Compute 
            # Bus time: Load Input + Load Next Weight
            
            t_pe = t_load_input_total + t_compute_total
            t_bus = t_load_input_total + t_load_next_weight
            
            # Actual time is the longer of the two
            actual_step_time = max(t_pe, t_bus)
            
            self.stats['cycles'] += actual_step_time
            self.stats['steady_cycles'] += actual_step_time
            self.stats['load_cycles'] += t_bus
            
            # Stall if weight load extends beyond compute
            if t_bus > t_pe:
                stall = t_bus - t_pe
                self.stats['stall_cycles'] += stall
            
            # -----------------------------------------------------------
            # Swap banks (weight[i+1] ready for next iteration)
            # -----------------------------------------------------------
            if next_c_tile < num_ch_tiles:
                self.w_sram.swap()
                print(f"  -> Swapped: weight[{next_c_tile}] now in compute bank")
        # ==========================================
        # EPILOGUE: Write output to DRAM
        # ==========================================
        
        print(f"[V2] Epilogue: Writing output...")
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
        print(f"[V2-FIXED] Finished in {end_time - start_time:.4f}s")
        
        return dram_out, self.stats
        
# ==========================================
# UTILITIES
# ==========================================

def load_data_strict(filename="benchmark_data.npz"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found!")
    
    print(f"\n[DATA] Loading: {filename}")
    data = np.load(filename)
    return data['d_in'], data['d_w']

def update_config_from_data(config, d_in, d_w):
    C_file, H_file, W_file = d_in.shape
    _, _, R_file, S_file = d_w.shape
    
    config.H = H_file
    config.W = W_file
    config.C = C_file
    config.R = R_file
    config.S = S_file

def validate_output(hw_out, d_in, d_w, config):
    print("\n[VALIDATION] Checking correctness...")
    
    H_out = config.H - config.R + 1
    W_out = config.W - config.S + 1
    
    if hw_out.shape != (H_out, W_out):
        print(f"❌ SIZE MISMATCH")
        return False

    errors = 0
    kernel = d_w[0]
    
    for oh in range(H_out):
        for ow in range(W_out):
            in_slice = d_in[:, oh:oh+config.R, ow:ow+config.S]
            sw_val = np.sum(in_slice * kernel)
            hw_val = hw_out[oh, ow]
            
            if hw_val != sw_val:
                errors += 1
                if errors <= 3:
                    print(f"  Error at [{oh},{ow}]: HW={hw_val}, SW={sw_val}")
    
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
    
    theo_weight = total_pixels * (config.C * config.R * config.S)
    theo_input = total_pixels * (config.C * config.R * config.S)
    
    actual_weight = stats['weight_dram_words']
    actual_input = stats['input_dram_words']
    
    w_reuse = theo_weight / actual_weight if actual_weight > 0 else 0
    i_reuse = theo_input / actual_input if actual_input > 0 else 0
    
    return w_reuse, i_reuse

def print_performance_report(stats, config):
    print("\n" + "="*70)
    print(f"  PERFORMANCE REPORT - V2 FIXED (WEIGHT STATIONARY)")
    print("="*70)
    
    total = stats['cycles']
    
    print(f"\n[CONFIGURATION]")
    print(f"  Architecture:     {config.PE_ROWS}×{config.PE_COLS} PEs")
    print(f"  Clock:            {config.FREQ_MHZ} MHz")
    print(f"  Dataflow:         Weight Stationary (Ping-Pong Overlap)")
    print(f"  Workload:         {config.H}×{config.W}×{config.C}, Kernel {config.R}×{config.S}")
    
    print(f"\n[MEMORY]")
    print(f"  On-chip SRAM:     {config.get_sram_size_kb():.2f} KB")
    print(f"    ├─ Weight Buf:  {2 * config.PE_ROWS * config.R * config.S * 4 / 1024:.2f} KB (ping-pong)")
    print(f"    ├─ Input Buf:   {2 * config.PE_ROWS * config.R * 18 * 4 / 1024:.2f} KB (ping-pong)")
    print(f"    └─ Global Acc:  {(config.H-config.R+1)*(config.W-config.S+1)*4/1024:.2f} KB")
    
    print(f"  DRAM Traffic:     {stats['dram_words']*4/1024:.2f} KB")
    print(f"    ├─ Weights:     {stats['weight_dram_words']*4/1024:.2f} KB")
    print(f"    ├─ Inputs:      {stats['input_dram_words']*4/1024:.2f} KB")
    print(f"    └─ Outputs:     {stats['output_dram_words']*4/1024:.2f} KB")
    
    print(f"\n[ACCESS COUNTS]")
    print(f"  Weight loads:     {stats['weight_loads']}")
    print(f"  Input loads:      {stats['input_loads']}")
    print(f"  Output writes:    {stats['output_writes']}")
    
    print(f"\n[TIMING - PIPELINE BREAKDOWN]")
    print(f"  Total cycles:     {total:,}")
    print(f"    ├─ Prologue:    {stats['prologue_cycles']:,} ({stats['prologue_cycles']/total*100:.1f}%)")
    print(f"    ├─ Steady:      {stats['steady_cycles']:,} ({stats['steady_cycles']/total*100:.1f}%)")
    print(f"    ├─ Epilogue:    {stats['epilogue_cycles']:,} ({stats['epilogue_cycles']/total*100:.1f}%)")
    print(f"    └─ Writeback:   {stats['writeback_cycles']:,} ({stats['writeback_cycles']/total*100:.1f}%)")
    
    print(f"\n[TIMING - COMPONENT BREAKDOWN]")
    print(f"  Compute cycles:   {stats['compute_cycles']:,} ({stats['compute_cycles']/total*100:.1f}%)")
    print(f"  Load cycles:      {stats['load_cycles']:,}")
    
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
    print(f"  Utilization:      {((total-stats['stall_cycles'])/total*100):.2f}%")
    print(f"  Throughput:       {gops:.2f} GOPS")
    
    w_reuse, i_reuse = calculate_reuse_factors(stats, config)
    print(f"\n[DATA REUSE]")
    print(f"  Weight reuse:     {w_reuse:.2f}x")
    print(f"  Input reuse:      {i_reuse:.2f}x")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    cfg = HardwareConfig()
    
    print("="*70)
    print("  V2 - WEIGHT STATIONARY WITH PING-PONG OVERLAP")
    print("="*70)
    
    try:
        d_in, d_w = load_data_strict("benchmark_data.npz")
        update_config_from_data(cfg, d_in, d_w)
        
        accel = HardwareAccelerator(cfg)
        hw_out, stats = accel.run(d_in, d_w)
        
        print_performance_report(stats, cfg)
        
        is_correct = validate_output(hw_out, d_in, d_w, cfg)
        
        if is_correct:
            print("✅ V2 SIMULATION SUCCESSFUL!")
        else:
            print("❌ V2 SIMULATION FAILED!")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()