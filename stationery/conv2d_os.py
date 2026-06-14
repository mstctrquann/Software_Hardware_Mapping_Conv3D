import sys, io
import numpy as np
import time

from convolution2d_mapping_baseline import HardwareConfig2D, SimulationStats, generate_data_2d, software_conv2d
from hardware_components import PingPongSRAM, PEArray, MemoryController

class Accelerator2D_OS:
    """Mô phỏng Output Stationary (OS) qua Object-Level FSM"""
    def __init__(self, config: HardwareConfig2D):
        self.cfg = config
        self.stats = SimulationStats()
        
        self.pe_array = PEArray(self.cfg.PE_ROWS, self.cfg.PE_COLS)
        self.sram_weight = PingPongSRAM("Weight_Buffer")
        self.sram_input = PingPongSRAM("Input_Buffer")
        self.mem_ctrl = MemoryController(self.cfg, self.stats)

    def run(self, dram_in, dram_w, verbose=False):
        print(f"[OS] Bắt đầu mô phỏng Structural Output Stationary cho Conv2D...")
        start_time = time.time()
        
        H_out = self.cfg.H - self.cfg.K_H + 1
        W_out = self.cfg.W - self.cfg.K_W + 1
        
        dram_out = np.zeros((self.cfg.M, H_out, W_out), dtype=np.int32)
        
        num_ch_tiles = self.cfg.C_in // self.cfg.PE_ROWS
        if self.cfg.C_in % self.cfg.PE_ROWS != 0:
            num_ch_tiles += 1
            
        self.stats.total_mac = self.cfg.M * H_out * W_out * self.cfg.C_in * self.cfg.K_H * self.cfg.K_W
        
        # OUTPUT STATIONARY FSM
        for m in range(self.cfg.M):
            for oh in range(H_out):
                for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                    valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                    
                    # Cài đặt Accumulator Register rỗng ban đầu
                    self.pe_array.reset_accumulator()
                    
                    for k in range(num_ch_tiles):
                        c_start = k * self.cfg.PE_ROWS
                        actual_channels = min(self.cfg.PE_ROWS, self.cfg.C_in - c_start)
                        
                        # ---------------------------------------------------------
                        # PHASE A: Stream Weights and Inputs simultaneously
                        # ---------------------------------------------------------
                        w_size = actual_channels * self.cfg.K_H * self.cfg.K_W
                        in_size = actual_channels * self.cfg.K_H * (valid_width + self.cfg.K_W - 1)
                        
                        t_load_w = self.mem_ctrl.fetch_weight_dram(w_size)
                        t_load_in = self.mem_ctrl.fetch_input_dram(in_size)
                        
                        w_slice = dram_w[m, c_start:c_start+actual_channels, :, :]
                        in_slice = dram_in[c_start:c_start+actual_channels, oh:oh+self.cfg.K_H, ow_tile:ow_tile+valid_width+self.cfg.K_W-1]
                        
                        self.sram_weight.load_from_dram(w_slice)
                        self.sram_input.load_from_dram(in_slice)
                        
                        self.sram_weight.swap()
                        self.sram_input.swap()
                        
                        w_bank = self.sram_weight.read_to_pe()
                        in_bank = self.sram_input.read_to_pe()
                        
                        macs_in_tile = actual_channels * valid_width * self.cfg.K_H * self.cfg.K_W
                        
                        self.mem_ctrl.access_sram_weight(macs_in_tile)
                        self.mem_ctrl.access_sram_input(macs_in_tile)
                        # KHÔNG access psum dram hay sram vì psum đứng yên trong Accumulator
                        
                        t_compute = self.mem_ctrl.compute_cycles(self.cfg.K_H * self.cfg.K_W)
                        
                        # --- Compute ---
                        for kh in range(self.cfg.K_H):
                            for kw in range(self.cfg.K_W):
                                # MẠCH ĐIỀU KHIỂN FSM OS:
                                # Weight và Input STREAM LIÊN TỤC
                                # Accumulator GIỮ NGUYÊN (Stationary) cộng dồn
                                self.pe_array.load_weight(w_bank[:, kh, kw].reshape(actual_channels, 1))
                                self.pe_array.load_input(in_bank[:, kh, kw : kw+valid_width])
                                self.pe_array.compute_mac()
                                
                        t_store_psum = 0
                        is_final = (k == num_ch_tiles - 1)
                        if is_final:
                            # Read out from PE Accumulator only at the very end
                            psum_out = self.pe_array.get_accumulator()
                            dram_out[m, oh, ow_tile:ow_tile+valid_width] = psum_out
                            
                            t_store_psum = self.mem_ctrl.store_psum_dram(valid_width, is_final=True)
                            self.mem_ctrl.stats.partial_sum_reads += valid_width # Extract from PE reg
                            
                        self.mem_ctrl.commit_pipeline_step(t_load_w + t_load_in, t_compute, t_store_psum)
                        
        end_time = time.time()
        print(f"[OS] Hoàn thành trong {end_time - start_time:.4f}s")
        return dram_out, self.stats

if __name__ == "__main__":
    cfg = HardwareConfig2D(H=16, W=16, C_in=32, M=4)
    d_in, d_w = generate_data_2d(cfg)
    accel = Accelerator2D_OS(cfg)
    hw_out, stats = accel.run(d_in, d_w)
    sw_out = software_conv2d(d_in, d_w, cfg)
    if np.array_equal(sw_out, hw_out):
        print("✅ OS Match!")
