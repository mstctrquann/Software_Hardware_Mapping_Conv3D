import sys, io
if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass
import numpy as np
import time

from convolution3d_mapping_baseline import HardwareConfig3D, SimulationStats, generate_data_3d, software_conv3d

class Accelerator3D_OS:
    """Mô phỏng Output Stationary (OS) Dataflow"""
    def __init__(self, config: HardwareConfig3D):
        self.cfg = config
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

    def run(self, dram_in, dram_w, verbose=False):
        print(f"[OS] Bắt đầu mô phỏng Output Stationary cho Conv3D...")
        start_time = time.time()
        
        D_out = self.cfg.D - self.cfg.K_D + 1
        H_out = self.cfg.H - self.cfg.K_H + 1
        W_out = self.cfg.W - self.cfg.K_W + 1
        
        dram_out = np.zeros((1, D_out, H_out, W_out), dtype=np.int32)
        
        num_ch_tiles = self.cfg.C_in // self.cfg.PE_ROWS
        if self.cfg.C_in % self.cfg.PE_ROWS != 0:
            num_ch_tiles += 1
            
        self.stats.total_mac = D_out * H_out * W_out * self.cfg.C_in * self.cfg.K_D * self.cfg.K_H * self.cfg.K_W
        
        # OUTPUT STATIONARY LOOP ORDER: Outer loops are Spatial (Outputs kept stationary in Accumulators)
        for od in range(D_out):
            for oh in range(H_out):
                for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                    valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                    
                    # Accumulator được reset tại đây
                    # Mọi giá trị cộng dồn sẽ nằm hoàn toàn trong PE Accumulator Registers
                    
                    for k in range(num_ch_tiles):
                        c_start = k * self.cfg.PE_ROWS
                        actual_channels = min(self.cfg.PE_ROWS, self.cfg.C_in - c_start)
                        
                        # Load Input Tile từ DRAM
                        in_size = actual_channels * self.cfg.K_D * self.cfg.K_H * (valid_width + self.cfg.K_W - 1)
                        t_load_in = self.load_input_from_dram(in_size)
                        
                        # Load Weight từ DRAM
                        w_size = actual_channels * self.cfg.K_D * self.cfg.K_H * self.cfg.K_W
                        t_load_w = self.load_weight_from_dram(w_size)
                        
                        # Compute
                        t_compute = self.cfg.K_D * self.cfg.K_H * self.cfg.K_W * self.cfg.CYCLE_MAC
                        self.stats.compute_cycles += t_compute
                        
                        # Track SRAM Reads
                        macs_in_tile = actual_channels * valid_width * self.cfg.K_D * self.cfg.K_H * self.cfg.K_W
                        # OS stream cả Weight và Input từ SRAM vào PE cho mỗi MAC
                        self.stats.sram_input_reads += macs_in_tile
                        self.stats.sram_weight_reads += macs_in_tile
                        
                        # Không có partial_sum_reads vì nó được giữ ở Accumulator trong PE
                        
                        for kd in range(self.cfg.K_D):
                            for kh in range(self.cfg.K_H):
                                for kw in range(self.cfg.K_W):
                                    w_vec = dram_w[0, c_start:c_start+actual_channels, kd, kh, kw].reshape(actual_channels, 1)
                                    in_mat = dram_in[c_start:c_start+actual_channels, od+kd, oh+kh, ow_tile+kw : ow_tile+kw+valid_width]
                                    partial_sum = np.sum(w_vec * in_mat, axis=0)
                                    dram_out[0, od, oh, ow_tile:ow_tile+valid_width] += partial_sum
                                    
                        t_store_psum = 0
                        # Cuối channel loop, ghi kết quả hoàn chỉnh ra DRAM
                        if k == num_ch_tiles - 1:
                            t_store_psum = self.store_psum_to_dram(valid_width, is_final=True)
                            
                        # Pipeline modeling
                        t_step = max(t_load_w + t_load_in, t_compute) + t_store_psum
                        self.stats.total_cycles += t_step
                        self.stats.stall_cycles += max(0, (t_load_w + t_load_in) - t_compute)
                        
        end_time = time.time()
        print(f"[OS] Hoàn thành trong {end_time - start_time:.4f}s")
        return dram_out, self.stats

if __name__ == "__main__":
    cfg = HardwareConfig3D(D=8, H=16, W=16, C_in=32)
    d_in, d_w = generate_data_3d(cfg)
    accel = Accelerator3D_OS(cfg)
    hw_out, stats = accel.run(d_in, d_w)
    sw_out = software_conv3d(d_in, d_w, cfg)
    
    stats.print_report("Output Stationary (OS)", cfg.PE_ROWS, cfg.PE_COLS)
    
    if np.array_equal(sw_out, hw_out):
        print("✅ OS Match!")
