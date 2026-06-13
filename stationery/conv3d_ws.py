import sys, io
if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass
import numpy as np
import time

from convolution3d_mapping_baseline import HardwareConfig3D, SimulationStats, generate_data_3d, software_conv3d

class PingPongSRAM3D_WS:
    def __init__(self, shape):
        self.buffer = np.zeros(shape, dtype=np.int32)
    def write_from_dram(self, data):
        # Mô phỏng thời gian copy được tính ở ngoài
        pass
    def read_to_pe(self):
        pass

class Accelerator3D_WS:
    """Mô phỏng Weight Stationary (WS) Dataflow"""
    def __init__(self, config: HardwareConfig3D):
        self.cfg = config
        self.stats = SimulationStats()
        
    def load_weight_from_dram(self, w_size):
        self.stats.dram_weight_reads += w_size
        return w_size * self.cfg.CYCLE_DRAM_RD
        
    def load_input_from_dram(self, in_size):
        self.stats.dram_input_reads += in_size
        return in_size * self.cfg.CYCLE_DRAM_RD
        
    def load_psum_from_dram(self, psum_size):
        self.stats.partial_sum_reads += psum_size
        return psum_size * self.cfg.CYCLE_DRAM_RD
        
    def store_psum_to_dram(self, psum_size, is_final=False):
        self.stats.partial_sum_writes += psum_size
        if is_final:
            self.stats.dram_output_writes += psum_size
        return psum_size * self.cfg.CYCLE_DRAM_RD

    def run(self, dram_in, dram_w, verbose=False):
        print(f"[WS] Bắt đầu mô phỏng Weight Stationary cho Conv3D...")
        start_time = time.time()
        
        D_out = self.cfg.D - self.cfg.K_D + 1
        H_out = self.cfg.H - self.cfg.K_H + 1
        W_out = self.cfg.W - self.cfg.K_W + 1
        
        dram_out = np.zeros((1, D_out, H_out, W_out), dtype=np.int32)
        
        num_ch_tiles = self.cfg.C_in // self.cfg.PE_ROWS
        if self.cfg.C_in % self.cfg.PE_ROWS != 0:
            num_ch_tiles += 1
            
        self.stats.total_mac = D_out * H_out * W_out * self.cfg.C_in * self.cfg.K_D * self.cfg.K_H * self.cfg.K_W
        
        # WEIGHT STATIONARY LOOP ORDER: Outer loop is Channel (Weights kept stationary)
        for k in range(num_ch_tiles):
            c_start = k * self.cfg.PE_ROWS
            actual_channels = min(self.cfg.PE_ROWS, self.cfg.C_in - c_start)
            
            # 1. Load Weight vào PE Register (Stationary qua toàn bộ Spatial Map)
            w_size = actual_channels * self.cfg.K_D * self.cfg.K_H * self.cfg.K_W
            t_load_w = self.load_weight_from_dram(w_size)
            
            # 2. Loop over Spatial Outputs (Stream Inputs qua PEs)
            for od in range(D_out):
                for oh in range(H_out):
                    for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                        valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                        
                        # Load Input Tile từ DRAM
                        in_size = actual_channels * self.cfg.K_D * self.cfg.K_H * (valid_width + self.cfg.K_W - 1)
                        t_load_in = self.load_input_from_dram(in_size)
                        
                        # Load Psum từ DRAM (Read-Modify-Write)
                        t_load_psum = 0
                        if k > 0:
                            t_load_psum = self.load_psum_from_dram(valid_width)
                            
                        # Tính số phép toán MAC trong tile này
                        macs_in_tile = actual_channels * valid_width * self.cfg.K_D * self.cfg.K_H * self.cfg.K_W
                        
                        # Track SRAM reads
                        # Trong WS:
                        # - Weight chỉ nạp 1 lần vào PE Register từ đầu chu kỳ `k`, nên sram_weight_reads bằng với w_size nạp ban đầu?
                        # Hoặc hiểu theo chuẩn: SRAM -> PE (1 lần), sau đó từ Register -> MAC (nhiều lần). 
                        # SRAM weight reads = w_size (chỉ đọc 1 lần từ SRAM vào PE cho toàn bộ spatial loop)
                        # Ở đây ta cộng dồn vào lúc nạp w_size (vì nó chỉ nạp 1 lần từ SRAM). Ta sẽ nạp ở đây.
                        if od == 0 and oh == 0 and ow_tile == 0:
                            self.stats.sram_weight_reads += w_size
                            
                        # - Input được stream liên tục từ SRAM -> PE
                        self.stats.sram_input_reads += macs_in_tile
                        
                        # Compute Cycles
                        t_compute = self.cfg.K_D * self.cfg.K_H * self.cfg.K_W * self.cfg.CYCLE_MAC
                        self.stats.compute_cycles += t_compute
                        
                        # Mô phỏng tính toán thực tế
                        for kd in range(self.cfg.K_D):
                            for kh in range(self.cfg.K_H):
                                for kw in range(self.cfg.K_W):
                                    w_vec = dram_w[0, c_start:c_start+actual_channels, kd, kh, kw].reshape(actual_channels, 1)
                                    in_mat = dram_in[c_start:c_start+actual_channels, od+kd, oh+kh, ow_tile+kw : ow_tile+kw+valid_width]
                                    partial_sum = np.sum(w_vec * in_mat, axis=0)
                                    dram_out[0, od, oh, ow_tile:ow_tile+valid_width] += partial_sum
                                    
                        # Store Psum ra DRAM
                        is_final = (k == num_ch_tiles - 1)
                        t_store_psum = self.store_psum_to_dram(valid_width, is_final=is_final)
                        
                        # Pipeline modeling
                        t_step = max(t_load_w + t_load_in + t_load_psum, t_compute) + t_store_psum
                        self.stats.total_cycles += t_step
                        self.stats.stall_cycles += max(0, (t_load_w + t_load_in + t_load_psum) - t_compute)
                        
                        # t_load_w chỉ áp dụng cho bước đầu tiên của vòng spatial
                        t_load_w = 0 
                        
        end_time = time.time()
        print(f"[WS] Hoàn thành trong {end_time - start_time:.4f}s")
        return dram_out, self.stats

if __name__ == "__main__":
    cfg = HardwareConfig3D(D=8, H=16, W=16, C_in=32)
    d_in, d_w = generate_data_3d(cfg)
    accel = Accelerator3D_WS(cfg)
    hw_out, stats = accel.run(d_in, d_w)
    sw_out = software_conv3d(d_in, d_w, cfg)
    if np.array_equal(sw_out, hw_out):
        print("✅ WS Match!")
