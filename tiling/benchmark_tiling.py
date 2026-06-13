import os
import sys
import io
import time
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def generate_data(C, H, W, seed=42):
    np.random.seed(seed)
    d_in = np.random.randint(-10, 10, size=(C, H, W), dtype=np.int32)
    d_w = np.random.randint(-10, 10, size=(1, C, 3, 3), dtype=np.int32)
    return d_in, d_w

def main():
    print("[BENCHMARK TILING] Loading modules...")
    v1 = load_module("v1", "v1_os.py")
    v2 = load_module("v2", "v2_ws.py")
    v3 = load_module("v3", "v3_ws_linebuf.py")

    os.makedirs("result_tiling", exist_ok=True)
    
    configs = [
        (16, 8, 8),
        (16, 16, 16),
        (32, 8, 8),
        (32, 16, 16)
    ]
    
    results = []

    with open("result_tiling/config_notes.txt", "w", encoding="utf-8") as f:
        f.write("=== TILING BENCHMARK CONFIG ===\n")
        f.write("PE Array Fixed: 16x16\n")
        f.write("Data Width: 32 bits\n")
        f.write("Frequency: 200.0 MHz\n\n")

        for C, H, W in configs:
            config_name = f"C={C}_H={H}_W={W}"
            f.write(f"\n--- WORKLOAD: {config_name} ---\n")
            print(f"\n==========================================")
            print(f" TESTING WORKLOAD: {config_name}")
            print(f"==========================================")
            
            d_in, d_w = generate_data(C, H, W)
            
            cfg1 = v1.HardwareConfig(C=C, H=H, W=W, PE_ROWS=16, PE_COLS=16)
            cfg2 = v2.HardwareConfig(C=C, H=H, W=W, PE_ROWS=16, PE_COLS=16)
            cfg3 = v3.HardwareConfig(C=C, H=H, W=W, PE_ROWS=16, PE_COLS=16)

            accel1 = v1.HardwareAccelerator(cfg1)
            hw_out1, stats1 = accel1.run(d_in, d_w, verbose=False)
            sram1 = cfg1.get_sram_size_kb()

            accel2 = v2.HardwareAccelerator(cfg2)
            hw_out2, stats2 = accel2.run(d_in, d_w, verbose=False)
            sram2 = cfg2.get_sram_size_kb() + (cfg2.H - cfg2.R + 1)*(cfg2.W - cfg2.S + 1)*4/1024

            accel3 = v3.HardwareAccelerator(cfg3)
            hw_out3, stats3 = accel3.run(d_in, d_w, verbose=False)
            line_buf_sram = (cfg3.PE_ROWS * cfg3.R * cfg3.W * 4) / 1024
            sram3 = cfg3.get_sram_size_kb() + (cfg3.H - cfg3.R + 1)*(cfg3.W - cfg3.S + 1)*4/1024 + line_buf_sram

            metrics = {
                "V1": {"stats": stats1, "sram": sram1},
                "V2": {"stats": stats2, "sram": sram2},
                "V3": {"stats": stats3, "sram": sram3},
            }
            results.append({"config": config_name, "metrics": metrics})

            f.write(f"[V1 Output Stationary]\nSRAM: {sram1:.2f} KB | Cycles: {stats1['cycles']} | Compute: {stats1['compute_cycles']} | Load time(cycles): {stats1['load_cycles']}\n")
            f.write(f"[V2 Weight Stationary]\nSRAM: {sram2:.2f} KB | Cycles: {stats2['cycles']} | Compute: {stats2['compute_cycles']} | Load time(cycles): {stats2['load_cycles']}\n")
            f.write(f"[V3 WS + Line Buffer]\nSRAM: {sram3:.2f} KB | Cycles: {stats3['cycles']} | Compute: {stats3['compute_cycles']} | Load time(cycles): {stats3['load_cycles']}\n")

    print("\n[BENCHMARK TILING] Generating Plots...")
    
    x = np.arange(len(configs))
    width = 0.25
    config_labels = [r["config"] for r in results]

    def plot_metric(metric_func, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        v1_data = [metric_func(r["metrics"]["V1"]) for r in results]
        v2_data = [metric_func(r["metrics"]["V2"]) for r in results]
        v3_data = [metric_func(r["metrics"]["V3"]) for r in results]

        b1 = plt.bar(x - width, v1_data, width, label='V1 (OS)')
        b2 = plt.bar(x, v2_data, width, label='V2 (WS)')
        b3 = plt.bar(x + width, v3_data, width, label='V3 (WS + LineBuf)')
        
        plt.bar_label(b1, padding=3, fmt='%.1f' if isinstance(v1_data[0], float) else '%d')
        plt.bar_label(b2, padding=3, fmt='%.1f' if isinstance(v2_data[0], float) else '%d')
        plt.bar_label(b3, padding=3, fmt='%.1f' if isinstance(v3_data[0], float) else '%d')

        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x, config_labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"result_tiling/{filename}.png")
        plt.close()

    plot_metric(lambda m: m["sram"], 'SRAM Capacity Comparison (KB)', 'SRAM (KB)', 'SRAM_Usage')
    plot_metric(lambda m: m["stats"]["cycles"], 'Total Cycles Comparison', 'Cycles', 'Total_Cycles')
    plot_metric(lambda m: m["stats"]["load_cycles"], 'Load Cycles Comparison', 'Load Cycles', 'Load_Cycles')
    plot_metric(lambda m: m["stats"]["dram_words"], 'DRAM Traffic (Words) Comparison', 'DRAM Words', 'DRAM_Traffic')

    idx = 3
    plt.figure(figsize=(12, 6))
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
    
    plt.yscale('log')
    plt.ylabel('Count (Log Scale)')
    plt.title(f'Compute and Access Counts (Config: {config_labels[idx]})')
    plt.xticks(x_cats, cats)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result_tiling/Counts_Comparison.png")
    plt.close()

    print("[BENCHMARK TILING] Done! Results saved to 'result_tiling' folder.")

if __name__ == "__main__":
    main()
