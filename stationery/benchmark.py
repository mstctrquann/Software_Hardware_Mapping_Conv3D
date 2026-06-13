import os
import sys
import io
import importlib.util
import time
import numpy as np
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def main():
    print("[BENCHMARK] Loading modules...")
    v1 = load_module("v1", "convolution_mapping_baseline(os).py")
    v2 = load_module("v2", "convolution_mapping_w_stationary.py")
    v3 = load_module("v3", "convolution_mapping_w_input.py")

    os.makedirs("Results", exist_ok=True)
    
    # We will test a few PE array sizes (User's scripts are hardcoded to 16 rows)
    pe_sizes = [(16, 16)]
    versions = ["V1 (Output Stat)", "V2 (Weight Stat)", "V3 (WS + LineBuf)"]
    
    results = []

    # Make sure data is generated first by V1
    print("[BENCHMARK] Generating/Loading common data...")
    cfg_base = v1.HardwareConfig()
    d_in, d_w = v1.load_or_generate_data(cfg_base, "benchmark_data.npz")

    with open("Results/config_notes.txt", "w", encoding="utf-8") as f:
        f.write("=== CONVOLUTION BLOCK BENCHMARK CONFIG ===\n")
        f.write(f"Input Shape (C, H, W): ({cfg_base.C}, {cfg_base.H}, {cfg_base.W})\n")
        f.write(f"Weight Shape (K, C, R, S): (1, {cfg_base.C}, {cfg_base.R}, {cfg_base.S})\n")
        f.write(f"Data Width: {cfg_base.DATA_WIDTH} bits\n")
        f.write(f"Frequency: {cfg_base.FREQ_MHZ} MHz\n\n")

        for pe_r, pe_c in pe_sizes:
            f.write(f"\n--- TILING CONFIG: PE_ROWS={pe_r}, PE_COLS={pe_c} ---\n")
            print(f"\n==========================================")
            print(f" TESTING PE ARRAY: {pe_r}x{pe_c}")
            print(f"==========================================")
            
            # Setup configs
            cfg1 = v1.HardwareConfig(PE_ROWS=pe_r, PE_COLS=pe_c)
            cfg2 = v2.HardwareConfig(PE_ROWS=pe_r, PE_COLS=pe_c)
            cfg3 = v3.HardwareConfig(PE_ROWS=pe_r, PE_COLS=pe_c)

            v2.update_config_from_data(cfg2, d_in, d_w)
            v3.update_config_from_data(cfg3, d_in, d_w)

            # --- RUN V1 ---
            accel1 = v1.HardwareAccelerator(cfg1)
            hw_out1, stats1 = accel1.run(d_in, d_w, verbose=False)
            sram1 = cfg1.get_sram_size_kb()

            # --- RUN V2 ---
            accel2 = v2.HardwareAccelerator(cfg2)
            hw_out2, stats2 = accel2.run(d_in, d_w, verbose=False)
            sram2 = cfg2.get_sram_size_kb()

            # --- RUN V3 ---
            accel3 = v3.HardwareAccelerator(cfg3)
            hw_out3, stats3 = accel3.run(d_in, d_w, verbose=False)
            sram3 = cfg3.get_sram_size_kb()

            metrics = {
                "V1": {"stats": stats1, "sram": sram1},
                "V2": {"stats": stats2, "sram": sram2},
                "V3": {"stats": stats3, "sram": sram3},
            }
            results.append({"pe": f"{pe_r}x{pe_c}", "metrics": metrics})

            f.write(f"\n[V1 Output Stationary]\nSRAM: {sram1:.2f} KB | Cycles: {stats1['cycles']} | Compute: {stats1['compute_cycles']} | Load time(cycles): {stats1['load_cycles']}\n")
            f.write(f"[V2 Weight Stationary]\nSRAM: {sram2:.2f} KB | Cycles: {stats2['cycles']} | Compute: {stats2['compute_cycles']} | Load time(cycles): {stats2['load_cycles']}\n")
            f.write(f"[V3 WS + Line Buffer]\nSRAM: {sram3:.2f} KB | Cycles: {stats3['cycles']} | Compute: {stats3['compute_cycles']} | Load time(cycles): {stats3['load_cycles']}\n")

    # Generate plots
    print("\n[BENCHMARK] Generating Plots...")
    
    # 1. SRAM Usage Comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(pe_sizes))
    width = 0.25

    sram_v1 = [r["metrics"]["V1"]["sram"] for r in results]
    sram_v2 = [r["metrics"]["V2"]["sram"] for r in results]
    sram_v3 = [r["metrics"]["V3"]["sram"] for r in results]

    b1 = plt.bar(x - width, sram_v1, width, label='V1 (OS)')
    b2 = plt.bar(x, sram_v2, width, label='V2 (WS)')
    b3 = plt.bar(x + width, sram_v3, width, label='V3 (WS + LineBuf)')
    plt.bar_label(b1, padding=3, fmt='%.2f')
    plt.bar_label(b2, padding=3, fmt='%.2f')
    plt.bar_label(b3, padding=3, fmt='%.2f')

    plt.ylabel('SRAM Capacity (KB)')
    plt.title('SRAM Capacity Comparison by Version')
    plt.xticks(x, [r["pe"] for r in results])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Results/SRAM_Usage.png")
    plt.close()

    # 2. Total Cycles Comparison
    plt.figure(figsize=(10, 6))
    cycles_v1 = [r["metrics"]["V1"]["stats"]["cycles"] for r in results]
    cycles_v2 = [r["metrics"]["V2"]["stats"]["cycles"] for r in results]
    cycles_v3 = [r["metrics"]["V3"]["stats"]["cycles"] for r in results]

    b1 = plt.bar(x - width, cycles_v1, width, label='V1 (OS)')
    b2 = plt.bar(x, cycles_v2, width, label='V2 (WS)')
    b3 = plt.bar(x + width, cycles_v3, width, label='V3 (WS + LineBuf)')
    plt.bar_label(b1, padding=3)
    plt.bar_label(b2, padding=3)
    plt.bar_label(b3, padding=3)

    plt.ylabel('Total Cycles')
    plt.title('Total Cycles Comparison')
    plt.xticks(x, [r["pe"] for r in results])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Results/Total_Cycles.png")
    plt.close()

    # 3. Load Time (Cycles) Comparison
    plt.figure(figsize=(10, 6))
    load_v1 = [r["metrics"]["V1"]["stats"]["load_cycles"] for r in results]
    load_v2 = [r["metrics"]["V2"]["stats"]["load_cycles"] for r in results]
    load_v3 = [r["metrics"]["V3"]["stats"]["load_cycles"] for r in results]

    b1 = plt.bar(x - width, load_v1, width, label='V1 (OS)')
    b2 = plt.bar(x, load_v2, width, label='V2 (WS)')
    b3 = plt.bar(x + width, load_v3, width, label='V3 (WS + LineBuf)')
    plt.bar_label(b1, padding=3)
    plt.bar_label(b2, padding=3)
    plt.bar_label(b3, padding=3)

    plt.ylabel('Load Cycles (DRAM Time)')
    plt.title('Load Cycles Comparison')
    plt.xticks(x, [r["pe"] for r in results])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Results/Load_Time.png")
    plt.close()
    
    # 4. Memory Accesses (Number of DRAM Words)
    plt.figure(figsize=(10, 6))
    dram_v1 = [r["metrics"]["V1"]["stats"]["dram_words"] for r in results]
    dram_v2 = [r["metrics"]["V2"]["stats"]["dram_words"] for r in results]
    dram_v3 = [r["metrics"]["V3"]["stats"]["dram_words"] for r in results]

    b1 = plt.bar(x - width, dram_v1, width, label='V1 (OS)')
    b2 = plt.bar(x, dram_v2, width, label='V2 (WS)')
    b3 = plt.bar(x + width, dram_v3, width, label='V3 (WS + LineBuf)')
    plt.bar_label(b1, padding=3)
    plt.bar_label(b2, padding=3)
    plt.bar_label(b3, padding=3)

    plt.ylabel('DRAM Traffic (Words)')
    plt.title('Total DRAM Traffic Comparison')
    plt.xticks(x, [r["pe"] for r in results])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Results/Memory_Accesses.png")
    plt.close()

    # 5. Computes, Loads, Stores (For PE=16x16 only as an example)
    plt.figure(figsize=(12, 6))
    idx = 0 # PE=16x16 (which is the first and only element now)
    v1_s = results[idx]["metrics"]["V1"]["stats"]
    v2_s = results[idx]["metrics"]["V2"]["stats"]
    v3_s = results[idx]["metrics"]["V3"]["stats"]
    
    cats = ['Compute Cycles', 'Weight Loads', 'Input Loads', 'Output Stores']
    v1_data = [v1_s['compute_cycles'], v1_s['weight_loads'], v1_s['input_loads'], v1_s['output_writes']]
    v2_data = [v2_s['compute_cycles'], v2_s['weight_loads'], v2_s['input_loads'], v2_s['output_writes']]
    v3_data = [v3_s['compute_cycles'], v3_s['weight_loads'], v3_s['input_loads'], v3_s['output_writes']]
    
    x_cats = np.arange(len(cats))
    b1 = plt.bar(x_cats - width, v1_data, width, label='V1')
    b2 = plt.bar(x_cats, v2_data, width, label='V2')
    b3 = plt.bar(x_cats + width, v3_data, width, label='V3')
    plt.bar_label(b1, padding=3)
    plt.bar_label(b2, padding=3)
    plt.bar_label(b3, padding=3)
    
    plt.yscale('log') # Log scale is better because compute cycles are much larger than loads/stores
    plt.ylabel('Count (Log Scale)')
    plt.title('Compute and Access Counts (PE 16x16)')
    plt.xticks(x_cats, cats)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Results/Counts_Comparison_16x16.png")
    plt.close()

    print("[BENCHMARK] Done! Results saved to 'Results' folder.")

if __name__ == "__main__":
    main()
