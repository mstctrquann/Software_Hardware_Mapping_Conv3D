import sys, io
if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass
import numpy as np
import time
import os
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stationery'))
from convolution3d_mapping_baseline import SimulationStats, generate_data_3d, software_conv3d

@dataclass
class HardwareConfigConv3D_Tiling:
    D: int = 16   
    H: int = 32   
    W: int = 32   
    C_in: int = 64
    K_D: int = 3
    K_H: int = 3
    K_W: int = 3
    PE_ROWS: int = 16   
    PE_COLS: int = 16   
    DATA_WIDTH: int = 32
    FREQ_MHZ: float = 200.0
    CYCLE_MAC: int = 1        
    CYCLE_DRAM_RD: int = 1
    T_Z: int = 8  
    T_C: int = 16 
    MAX_SRAM_KB: float = 64.0 

    def get_required_sram_kb(self):
        w_bits = 2 * (self.T_C * self.K_D * self.K_H * self.K_W) * self.DATA_WIDTH
        in_d = self.T_Z + self.K_D - 1
        in_req_w = self.PE_COLS + self.K_W - 1
        in_bits = 2 * (self.T_C * in_d * self.K_H * in_req_w) * self.DATA_WIDTH
        return (w_bits + in_bits) / 8192

class Accelerator3D_Tiling:
    def __init__(self, config):
        self.cfg = config
        self.sram_req = config.get_required_sram_kb()
        
        print(f"[INIT] Required SRAM: {self.sram_req:.2f} KB / MAX: {config.MAX_SRAM_KB} KB")
        if self.sram_req > config.MAX_SRAM_KB:
            print("⚠️ CẢNH BÁO: Tiling size vượt quá dung lượng SRAM on-chip!")
            
        self.stats = SimulationStats()

    def load_weight_from_dram(self, w_size):
        self.stats.dram_weight_reads += w_size
        return w_size * self.cfg.CYCLE_DRAM_RD
        
    def load_input_from_dram(self, in_size):
        self.stats.dram_input_reads += in_size
        return in_size * self.cfg.CYCLE_DRAM_RD
        
    def store_psum_to_dram(self, psum_size, is_final=False):
        self.stats.partial_sum_writes += psum_size
        if is_final:
            self.stats.dram_output_writes += psum_size
        return psum_size * self.cfg.CYCLE_DRAM_RD

    def run(self, dram_in, dram_w):
        print(f"[TILING] Chạy mô phỏng Tiling Z={self.cfg.T_Z}, C={self.cfg.T_C}")
        start_time = time.time()
        
        D_out = self.cfg.D - self.cfg.K_D + 1
        H_out = self.cfg.H - self.cfg.K_H + 1  
        W_out = self.cfg.W - self.cfg.K_W + 1  
        
        dram_out = np.zeros((1, D_out, H_out, W_out), dtype=np.int32)
        
        num_z_tiles = int(np.ceil(D_out / self.cfg.T_Z))
        num_c_tiles = int(np.ceil(self.cfg.C_in / self.cfg.T_C))
        
        self.stats.total_mac = D_out * H_out * W_out * self.cfg.C_in * self.cfg.K_D * self.cfg.K_H * self.cfg.K_W
        
        for tz in range(num_z_tiles):
            z_start = tz * self.cfg.T_Z
            z_end = min(z_start + self.cfg.T_Z, D_out)
            
            for oh in range(H_out):
                for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                    
                    acc_buffer = np.zeros((z_end - z_start, self.cfg.PE_COLS), dtype=np.int32)
                    
                    for tc in range(num_c_tiles):
                        c_start = tc * self.cfg.T_C
                        c_end = min(c_start + self.cfg.T_C, self.cfg.C_in)
                        actual_tc = c_end - c_start
                        
                        w_size = actual_tc * self.cfg.K_D * self.cfg.K_H * self.cfg.K_W
                        valid_z_in = (z_end - z_start) + self.cfg.K_D - 1
                        in_req_w = self.cfg.PE_COLS + self.cfg.K_W - 1
                        in_size = actual_tc * valid_z_in * self.cfg.K_H * in_req_w
                        
                        t_load_w = self.load_weight_from_dram(w_size)
                        t_load_in = self.load_input_from_dram(in_size)
                        t_load_total = t_load_w + t_load_in
                        
                        t_compute_tile = 0
                        for z_idx, od in enumerate(range(z_start, z_end)):
                            valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                            
                            macs_in_z_slice = actual_tc * valid_width * self.cfg.K_D * self.cfg.K_H * self.cfg.K_W
                            self.stats.sram_input_reads += macs_in_z_slice
                            self.stats.sram_weight_reads += macs_in_z_slice
                            
                            for kd in range(self.cfg.K_D):
                                for kh in range(self.cfg.K_H):
                                    for kw in range(self.cfg.K_W):
                                        w_patch = dram_w[0, c_start:c_end, kd, kh, kw]
                                        in_patch = dram_in[c_start:c_end, od+kd, oh+kh, ow_tile+kw:ow_tile+kw+valid_width]
                                        for w_idx in range(valid_width):
                                            mac_val = np.sum(w_patch * in_patch[:, w_idx])
                                            acc_buffer[z_idx, w_idx] += mac_val
                                        
                                        t_compute_tile += self.cfg.CYCLE_MAC * actual_tc
                        
                        self.stats.compute_cycles += t_compute_tile
                        t_step = max(t_load_total, t_compute_tile)
                        self.stats.total_cycles += t_step
                        self.stats.stall_cycles += max(0, t_load_total - t_compute_tile)

                    for z_idx, od in enumerate(range(z_start, z_end)):
                        valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                        dram_out[0, od, oh, ow_tile:ow_tile+valid_width] = acc_buffer[z_idx, :valid_width]
                        t_write = self.store_psum_to_dram(valid_width, is_final=True)
                        self.stats.total_cycles += t_write
        
        end_time = time.time()
        print(f"[TILING] Hoàn thành trong {end_time - start_time:.4f}s")
        return dram_out, self.stats

if __name__ == "__main__":
    cfg = HardwareConfigConv3D_Tiling(T_Z=4, T_C=32)
    d_in, d_w = generate_data_3d(cfg)
    accel = Accelerator3D_Tiling(cfg)
    hw_out, stats = accel.run(d_in, d_w)
    sw_out = software_conv3d(d_in, d_w, cfg)
    if np.array_equal(sw_out, hw_out):
        print("✅ TILING Match!")
