import sys, io
if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass

import numpy as np

sys.path.append('./stationery')
sys.path.append('./tiling')

from convolution3d_mapping_baseline import HardwareConfig3D, generate_data_3d, software_conv3d
from conv3d_baseline import Accelerator3D_Baseline
from conv3d_os import Accelerator3D_OS
from conv3d_ws import Accelerator3D_WS
from conv3d_is import Accelerator3D_IS
from conv3d_tiling_os import Accelerator3D_Tiling, HardwareConfigConv3D_Tiling

def format_row(name, stats, cfg=None):
    if stats is None:
        return f"{name:<25} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}"
    pass

def run_benchmark():
    print("="*100)
    print("  BENCHMARKING CONV3D SIMULATORS (DATAFLOW EXPLORATION)")
    print("="*100)
    
    cfg_base = HardwareConfig3D(D=16, H=32, W=32, C_in=64)
    d_in, d_w = generate_data_3d(cfg_base)
    
    # Baseline
    accel_base = Accelerator3D_Baseline(cfg_base)
    _, stats_base = accel_base.run(d_in, d_w)
    
    # OS
    accel_os = Accelerator3D_OS(cfg_base)
    _, stats_os = accel_os.run(d_in, d_w)
    
    # WS
    accel_ws = Accelerator3D_WS(cfg_base)
    _, stats_ws = accel_ws.run(d_in, d_w)
    
    # IS
    accel_is = Accelerator3D_IS(cfg_base)
    _, stats_is = accel_is.run(d_in, d_w)
    
    # Tiling (OS with SRAM constraint)
    cfg_tiling = HardwareConfigConv3D_Tiling(D=16, H=32, W=32, C_in=64, T_Z=8, T_C=16, MAX_SRAM_KB=64.0)
    accel_tiling = Accelerator3D_Tiling(cfg_tiling)
    _, stats_tiling = accel_tiling.run(d_in, d_w)

    dataflows = [
        ("Baseline (No Stationary)", stats_base, cfg_base),
        ("Weight Stationary (WS)", stats_ws, cfg_base),
        ("Input Stationary (IS)", stats_is, cfg_base),
        ("Output Stationary (OS)", stats_os, cfg_base),
        ("Tiling OS (Z=8, C=16)", stats_tiling, cfg_tiling)
    ]

    print("\n" + "="*120)
    print(f"{'Dataflow':<25} | {'Total MACs':<15} | {'Compute Cycles':<15} | {'PE Utilization':<15} | {'MACs/Cycle':<15}")
    print("-" * 120)
    for name, st, cfg in dataflows:
        pe_util = f"{st.pe_utilization(cfg.PE_ROWS, cfg.PE_COLS)*100:.1f}%"
        thru = f"{st.throughput_mac_per_cycle():.2f}"
        print(f"{name:<25} | {st.total_mac:<15,} | {st.compute_cycles:<15,} | {pe_util:<15} | {thru:<15}")
    print("="*120)

    print("\n[LOAD METRICS]")
    print(f"{'Dataflow':<25} | {'DRAM W Reads':<15} | {'DRAM In Reads':<15} | {'SRAM W Reads':<15} | {'SRAM In Reads':<15}")
    print("-" * 120)
    for name, st, _ in dataflows:
        print(f"{name:<25} | {st.dram_weight_reads:<15,} | {st.dram_input_reads:<15,} | {st.sram_weight_reads:<15,} | {st.sram_input_reads:<15,}")
    print("="*120)

    print("\n[STORE METRICS]")
    print(f"{'Dataflow':<25} | {'PSum SRAM/DRAM Rd':<20} | {'PSum SRAM/DRAM Wr':<20} | {'DRAM Out Writes':<15}")
    print("-" * 120)
    for name, st, _ in dataflows:
        print(f"{name:<25} | {st.partial_sum_reads:<20,} | {st.partial_sum_writes:<20,} | {st.dram_output_writes:<15,}")
    print("="*120)

    print("\n[REUSE METRICS]")
    print(f"{'Dataflow':<25} | {'Weight Reuse':<15} | {'Input Reuse':<15} | {'Output/PSum Reuse':<18}")
    print("-" * 120)
    for name, st, _ in dataflows:
        w_re = f"{st.weight_reuse_factor():.2f}"
        i_re = f"{st.input_reuse_factor():.2f}"
        o_re = f"{st.output_reuse_factor():.2f}"
        if st.dram_weight_reads == 0: w_re = "INF"
        print(f"{name:<25} | {w_re:<15} | {i_re:<15} | {o_re:<18}")
    print("="*120)
    
    print("\n[CYCLES COMPARISON]")
    print(f"{'Dataflow':<25} | {'Total Cycles':<15} | {'Compute Cycles':<15} | {'Stall Cycles':<15}")
    print("-" * 120)
    for name, st, _ in dataflows:
        print(f"{name:<25} | {st.total_cycles:<15,} | {st.compute_cycles:<15,} | {st.stall_cycles:<15,}")
    print("="*120)

if __name__ == "__main__":
    run_benchmark()
