import sys, io
if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass

import numpy as np
import time
from dataclasses import dataclass
import os

@dataclass
class HardwareConfig3D:
    # ==========================================
    # 1. KÍCH THƯỚC BÀI TOÁN (WORKLOAD)
    # ==========================================
    D: int = 16     # Depth 
    H: int = 32     # Height
    W: int = 32     # Width
    C_in: int = 64  # Input Channels
    
    # Kích thước Kernel 3D
    K_D: int = 3
    K_H: int = 3
    K_W: int = 3

    # ==========================================
    # 2. KIẾN TRÚC PHẦN CỨNG (ARCHITECTURE)
    # ==========================================
    PE_ROWS: int = 16   # Số hàng PE (unroll input channels)
    PE_COLS: int = 16   # Số cột PE (unroll output width)
    DATA_WIDTH: int = 32  # 32-bit integer

    # ==========================================
    # 3. THÔNG SỐ THỜI GIAN (TIMING)
    # ==========================================
    FREQ_MHZ: float = 200.0
    CYCLE_MAC: int = 1        
    CYCLE_DRAM_RD: int = 1    

class SimulationStats:
    def __init__(self):
        self.total_cycles = 0
        self.compute_cycles = 0
        self.stall_cycles = 0
        self.total_mac = 0
        
        # Load Metrics
        self.dram_weight_reads = 0
        self.dram_input_reads = 0
        self.sram_weight_reads = 0
        self.sram_input_reads = 0
        
        # Store Metrics
        self.dram_output_writes = 0
        self.partial_sum_reads = 0
        self.partial_sum_writes = 0
        
    def pe_utilization(self, pe_rows, pe_cols):
        if self.compute_cycles == 0: return 0.0
        return self.total_mac / (self.compute_cycles * pe_rows * pe_cols)
        
    def throughput_mac_per_cycle(self):
        if self.total_cycles == 0: return 0.0
        return self.total_mac / self.total_cycles
        
    def weight_reuse_factor(self):
        if self.dram_weight_reads == 0: return 0.0
        return self.total_mac / self.dram_weight_reads
        
    def input_reuse_factor(self):
        if self.dram_input_reads == 0: return 0.0
        return self.total_mac / self.dram_input_reads
        
    def output_reuse_factor(self):
        # Depending on whether we measure reuse as MACs / output writes
        # OS keeps PSUMs inside PE accumulator (no SRAM write), WS reads/writes to SRAM/DRAM.
        # But specifically, the README says: "OS primary reuse target: Partial Sum".
        # Reuse = MACs / (partial_sum_reads + partial_sum_writes + dram_output_writes).
        total_output_traffic = self.partial_sum_reads + self.partial_sum_writes + self.dram_output_writes
        if total_output_traffic == 0: return 0.0
        return self.total_mac / total_output_traffic
        
    def print_report(self, name, pe_rows, pe_cols):
        print(f"\n{'='*70}")
        print(f"  PERFORMANCE REPORT - {name.upper()}")
        print(f"{'='*70}")
        
        print(f"\n[COMPUTE METRICS]")
        print(f"  Total MACs:       {self.total_mac:,}")
        print(f"  Compute Cycles:   {self.compute_cycles:,}")
        print(f"  PE Utilization:   {self.pe_utilization(pe_rows, pe_cols)*100:.1f}%")
        print(f"  MACs/Cycle:       {self.throughput_mac_per_cycle():.2f}")
        
        print(f"\n[LOAD METRICS]")
        print(f"  DRAM W Reads:     {self.dram_weight_reads:,}")
        print(f"  DRAM In Reads:    {self.dram_input_reads:,}")
        print(f"  SRAM W Reads:     {self.sram_weight_reads:,}")
        print(f"  SRAM In Reads:    {self.sram_input_reads:,}")
        
        print(f"\n[STORE METRICS]")
        print(f"  PSum SRAM/DRAM Rd:{self.partial_sum_reads:,}")
        print(f"  PSum SRAM/DRAM Wr:{self.partial_sum_writes:,}")
        print(f"  DRAM Out Writes:  {self.dram_output_writes:,}")
        
        print(f"\n[REUSE METRICS]")
        w_re = f"{self.weight_reuse_factor():.2f}" if self.dram_weight_reads > 0 else "INF"
        print(f"  Weight Reuse:     {w_re}")
        print(f"  Input Reuse:      {self.input_reuse_factor():.2f}")
        print(f"  Output Reuse:     {self.output_reuse_factor():.2f}")
        
        print(f"\n[CYCLES COMPARISON]")
        print(f"  Total Cycles:     {self.total_cycles:,}")
        print(f"  Stall Cycles:     {self.stall_cycles:,}")
        print(f"{'='*70}\n")

def generate_data_3d(config):
    """Generate mock data for Conv3D validation"""
    print(f"[DATA] Generating random data for Conv3D...")
    # Dữ liệu Input: (C_in, D, H, W)
    d_in = np.random.randint(0, 5, (config.C_in, config.D, config.H, config.W), dtype=np.int32)
    # Trọng số: (C_out=1, C_in, K_D, K_H, K_W)
    d_w = np.random.randint(0, 5, (1, config.C_in, config.K_D, config.K_H, config.K_W), dtype=np.int32)
    
    return d_in, d_w

def software_conv3d(d_in, d_w, config):
    """Tính toán Conv3D tuần tự trên CPU để làm baseline đối chiếu kết quả (Golden Model)"""
    D_out = config.D - config.K_D + 1
    H_out = config.H - config.K_H + 1
    W_out = config.W - config.K_W + 1
    
    out = np.zeros((1, D_out, H_out, W_out), dtype=np.int32) # (C_out=1, D_out, H_out, W_out)
    
    total_macs = 0
    print("[SW BASELINE] Bắt đầu tính toán Conv3D thuần túy...")
    start_time = time.time()
    
    # 6 vòng lặp for (Loop Nest) tương ứng với Conv3D
    for od in range(D_out):
        for oh in range(H_out):
            for ow in range(W_out):
                # Trích xuất khối voxel 3D (patch) từ Input Feature Map
                in_patch = d_in[:, od:od+config.K_D, oh:oh+config.K_H, ow:ow+config.K_W]
                # Thực hiện MAC (Multiply-Accumulate) với duy nhất 1 Kernel (c_out = 0)
                out[0, od, oh, ow] = np.sum(in_patch * d_w[0])
                total_macs += config.C_in * config.K_D * config.K_H * config.K_W
                
    end_time = time.time()
    
    print("="*50)
    print(f"  [SW BASELINE] REPORT")
    print("="*50)
    print(f"Thời gian chạy:      {end_time - start_time:.4f}s")
    print(f"Kích thước Output:   {out.shape}")
    print(f"Tổng số phép MAC:    {total_macs:,}")
    print("="*50)
    return out

if __name__ == "__main__":
    cfg = HardwareConfig3D()
    d_in, d_w = generate_data_3d(cfg)
    golden_out = software_conv3d(d_in, d_w, cfg)
