import sys, io
import numpy as np
import time

from convolution2d_mapping_baseline import HardwareConfig2D, SimulationStats, generate_data_2d, software_conv2d
from hardware_components import PingPongSRAM, PEArray, MemoryController

class Accelerator2D_IS:
    """Mô phỏng Input Stationary (IS) qua Object-Level FSM"""
    def __init__(self, config: HardwareConfig2D):
        self.cfg = config
        self.stats = SimulationStats()
        
        self.pe_array = PEArray(self.cfg.PE_ROWS, self.cfg.PE_COLS)
        self.sram_weight = PingPongSRAM("Weight_Buffer")
        self.sram_input = PingPongSRAM("Input_Buffer")
        self.mem_ctrl = MemoryController(self.cfg, self.stats)

    def run(self, dram_in, dram_w, verbose=False):
        print(f"[IS] Bắt đầu mô phỏng Structural Input Stationary cho Conv2D...")
        start_time = time.time()
        
        H_out = self.cfg.H - self.cfg.K_H + 1
        W_out = self.cfg.W - self.cfg.K_W + 1
        
        dram_out = np.zeros((self.cfg.M, H_out, W_out), dtype=np.int32)
        
        num_ch_tiles = self.cfg.C_in // self.cfg.PE_ROWS
        if self.cfg.C_in % self.cfg.PE_ROWS != 0:
            num_ch_tiles += 1
            
        self.stats.total_mac = self.cfg.M * H_out * W_out * self.cfg.C_in * self.cfg.K_H * self.cfg.K_W
        
        # INPUT STATIONARY FSM
        for oh in range(H_out):
            for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                
                for k in range(num_ch_tiles):
                    c_start = k * self.cfg.PE_ROWS
                    actual_channels = min(self.cfg.PE_ROWS, self.cfg.C_in - c_start)
                    
                    # ---------------------------------------------------------
                    # PHASE A: Load Input Tile into PingPong SRAM & PE Register
                    # ---------------------------------------------------------
                    in_size = actual_channels * self.cfg.K_H * (valid_width + self.cfg.K_W - 1)
                    t_load_in = self.mem_ctrl.fetch_input_dram(in_size)
                    
                    in_slice = dram_in[c_start:c_start+actual_channels, oh:oh+self.cfg.K_H, ow_tile:ow_tile+valid_width+self.cfg.K_W-1]
                    self.sram_input.load_from_dram(in_slice)
                    self.sram_input.swap()
                    in_bank = self.sram_input.read_to_pe()
                    
                    # ---------------------------------------------------------
                    # PHASE B: Stream Weights over stationary inputs
                    # ---------------------------------------------------------
                    for m in range(self.cfg.M):
                        w_size = actual_channels * self.cfg.K_H * self.cfg.K_W
                        t_load_w = self.mem_ctrl.fetch_weight_dram(w_size)
                        
                        w_slice = dram_w[m, c_start:c_start+actual_channels, :, :]
                        self.sram_weight.load_from_dram(w_slice)
                        
                        t_load_psum = 0
                        if k > 0:
                            t_load_psum = self.mem_ctrl.fetch_psum_dram(valid_width)
                            self.pe_array.load_psum(np.copy(dram_out[m, oh, ow_tile:ow_tile+valid_width]))
                        else:
                            self.pe_array.reset_accumulator()
                            
                        self.sram_weight.swap()
                        w_bank = self.sram_weight.read_to_pe()
                        
                        macs_in_tile = actual_channels * valid_width * self.cfg.K_H * self.cfg.K_W
                        
                        if m == 0:
                            self.mem_ctrl.access_sram_input(in_size) # SRAM -> PE Register read once per weight pass
                            
                        self.mem_ctrl.access_sram_weight(macs_in_tile)
                        self.mem_ctrl.access_psum_pe(macs_in_tile)
                        
                        t_compute = self.mem_ctrl.compute_cycles(self.cfg.K_H * self.cfg.K_W)
                        
                        # --- Compute ---
                        for kh in range(self.cfg.K_H):
                            for kw in range(self.cfg.K_W):
                                # MẠCH ĐIỀU KHIỂN FSM IS:
                                # Input được NẠP VÀO THANH GHI 1 LẦN VÀ GIỮ NGUYÊN (Khóa tĩnh)
                                # Weight được STREAM LIÊN TỤC
                                self.pe_array.load_weight(w_bank[:, kh, kw].reshape(actual_channels, 1))
                                self.pe_array.load_input(in_bank[:, kh, kw : kw+valid_width])
                                self.pe_array.compute_mac()
                                
                        # --- Store ---
                        psum_out = self.pe_array.get_accumulator()
                        dram_out[m, oh, ow_tile:ow_tile+valid_width] = psum_out
                        
                        is_final = (k == num_ch_tiles - 1)
                        t_store_psum = self.mem_ctrl.store_psum_dram(valid_width, is_final=is_final)
                        
                        self.mem_ctrl.commit_pipeline_step(t_load_w + t_load_in + t_load_psum, t_compute, t_store_psum)
                        
                        t_load_in = 0 # Input đã khóa trong register, không tốn latency cho kernel tiếp theo
                        
        end_time = time.time()
        print(f"[IS] Hoàn thành trong {end_time - start_time:.4f}s")
        return dram_out, self.stats

if __name__ == "__main__":
    cfg = HardwareConfig2D(H=16, W=16, C_in=32, M=4)
    d_in, d_w = generate_data_2d(cfg)
    accel = Accelerator2D_IS(cfg)
    hw_out, stats = accel.run(d_in, d_w)
    sw_out = software_conv2d(d_in, d_w, cfg)
    if np.array_equal(sw_out, hw_out):
        print("✅ IS Match!")
