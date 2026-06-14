import sys, io
import numpy as np
import time

from convolution2d_mapping_baseline import HardwareConfig2D, SimulationStats, generate_data_2d, software_conv2d
from hardware_components import PingPongSRAM, PEArray, MemoryController

class Accelerator2D_WS:
    """Mô phỏng True Weight Stationary (WS) qua Object-Level FSM"""
    def __init__(self, config: HardwareConfig2D):
        self.cfg = config
        self.stats = SimulationStats()
        
        self.pe_array = PEArray(self.cfg.PE_ROWS, self.cfg.PE_COLS)
        self.sram_weight = PingPongSRAM("Weight_Buffer")
        self.sram_input = PingPongSRAM("Input_Buffer")
        self.mem_ctrl = MemoryController(self.cfg, self.stats)

    def run(self, dram_in, dram_w, verbose=False):
        print(f"[WS] Bắt đầu mô phỏng True Structural Weight Stationary cho Conv2D...")
        start_time = time.time()
        
        H_out = self.cfg.H - self.cfg.K_H + 1
        W_out = self.cfg.W - self.cfg.K_W + 1
        
        dram_out = np.zeros((self.cfg.M, H_out, W_out), dtype=np.int32)
        
        num_ch_tiles = self.cfg.C_in // self.cfg.PE_ROWS
        if self.cfg.C_in % self.cfg.PE_ROWS != 0:
            num_ch_tiles += 1
            
        self.stats.total_mac = self.cfg.M * H_out * W_out * self.cfg.C_in * self.cfg.K_H * self.cfg.K_W
        
        # ---------------------------------------------------------
        # TRUE WEIGHT STATIONARY FSM
        # ---------------------------------------------------------
        # Trật tự vòng lặp đúng của WS:
        # 1. Output Channel (m)
        # 2. Input Channel Tile (k)
        # 3. Kernel H (kh)
        # 4. Kernel W (kw)
        # ---> KHÓA TĨNH TRỌNG SỐ TẠI ĐÂY <---
        # 5. Spatial H (oh)
        # 6. Spatial W (ow_tile)
        # ---> STREAM INPUT QUA <---
        
        for m in range(self.cfg.M):
            for k in range(num_ch_tiles):
                c_start = k * self.cfg.PE_ROWS
                actual_channels = min(self.cfg.PE_ROWS, self.cfg.C_in - c_start)
                
                # --- Nạp toàn bộ Kernel Weight từ DRAM vào PingPong SRAM ---
                w_size = actual_channels * self.cfg.K_H * self.cfg.K_W
                t_load_w_dram = self.mem_ctrl.fetch_weight_dram(w_size)
                
                w_slice = dram_w[m, c_start:c_start+actual_channels, :, :]
                self.sram_weight.load_from_dram(w_slice)
                self.sram_weight.swap()
                w_bank = self.sram_weight.read_to_pe()
                
                first_dram_pass = True
                
                for kh in range(self.cfg.K_H):
                    for kw in range(self.cfg.K_W):
                        
                        # Đo đạc cụ thể: Nạp Weight từ SRAM vào PE Register
                        w_elements = actual_channels
                        t_sram_w = self.mem_ctrl.access_sram_weight(w_elements)
                        
                        # Pipeline setup stall: Phải đợi SRAM nạp xong mới bắt đầu stream Input
                        self.stats.total_cycles += t_sram_w
                        self.stats.stall_cycles += t_sram_w
                        
                        # ---> MẠCH ĐIỀU KHIỂN FSM WS: KHÓA TĨNH TRỌNG SỐ <---
                        self.pe_array.load_weight(w_bank[:, kh, kw].reshape(actual_channels, 1))
                        
                        # --- Stream toàn bộ ảnh không gian qua trọng số tĩnh ---
                        for oh in range(H_out):
                            for ow_tile in range(0, W_out, self.cfg.PE_COLS):
                                valid_width = min(self.cfg.PE_COLS, W_out - ow_tile)
                                
                                # Background Load Input từ DRAM
                                in_size = actual_channels * valid_width
                                t_load_in_dram = self.mem_ctrl.fetch_input_dram(in_size)
                                
                                in_slice = dram_in[c_start:c_start+actual_channels, oh+kh:oh+kh+1, ow_tile+kw:ow_tile+kw+valid_width]
                                self.sram_input.load_from_dram(in_slice)
                                
                                # Đọc lại Partial Sum (Nhược điểm của WS: Tốn lưu lượng Psum)
                                t_load_psum = 0
                                if k > 0 or kh > 0 or kw > 0:
                                    t_load_psum = self.mem_ctrl.fetch_psum_dram(valid_width)
                                    self.pe_array.load_psum(np.copy(dram_out[m, oh, ow_tile:ow_tile+valid_width]))
                                else:
                                    self.pe_array.reset_accumulator()
                                    
                                self.sram_input.swap()
                                in_bank = self.sram_input.read_to_pe()
                                
                                # PE đọc Input từ SRAM
                                self.mem_ctrl.access_sram_input(in_size)
                                self.mem_ctrl.access_psum_pe(actual_channels * valid_width)
                                
                                t_compute = self.mem_ctrl.compute_cycles(1) # Chỉ tính 1 phần tử MAC
                                
                                # ---> MẠCH ĐIỀU KHIỂN FSM WS: STREAM INPUT <---
                                self.pe_array.load_input(in_bank[:, 0, :])
                                self.pe_array.compute_mac()
                                
                                # Lưu Psum ra ngoài
                                psum_out = self.pe_array.get_accumulator()
                                dram_out[m, oh, ow_tile:ow_tile+valid_width] = psum_out
                                
                                is_final = (k == num_ch_tiles - 1 and kh == self.cfg.K_H - 1 and kw == self.cfg.K_W - 1)
                                t_store_psum = self.mem_ctrl.store_psum_dram(valid_width, is_final=is_final)
                                
                                # Xử lý Pipeline Timing tổng
                                t_background_load = t_load_in_dram + t_load_psum
                                if first_dram_pass:
                                    t_background_load += t_load_w_dram
                                    first_dram_pass = False
                                    
                                self.mem_ctrl.commit_pipeline_step(t_background_load, t_compute, t_store_psum)
                                
        end_time = time.time()
        print(f"[WS] Hoàn thành trong {end_time - start_time:.4f}s")
        return dram_out, self.stats

if __name__ == "__main__":
    cfg = HardwareConfig2D(H=16, W=16, C_in=32, M=4)
    d_in, d_w = generate_data_2d(cfg)
    accel = Accelerator2D_WS(cfg)
    hw_out, stats = accel.run(d_in, d_w)
    sw_out = software_conv2d(d_in, d_w, cfg)
    if np.array_equal(sw_out, hw_out):
        print("✅ WS Match!")
