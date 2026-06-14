import numpy as np

class PingPongSRAM:
    """
    Mô phỏng bộ đệm Ping-Pong kép (Double Buffering).
    Cho phép đọc (foreground) và nạp (background) xảy ra song song.
    """
    def __init__(self, name):
        self.name = name
        self.bank_ping = None
        self.bank_pong = None
        self.state = "PING" # "PING" active: PE reads Ping, DRAM writes Pong.
        
    def load_from_dram(self, data):
        """DRAM nạp dữ liệu vào bank đang chạy nền (Background)"""
        if self.state == "PING":
            self.bank_pong = data
        else:
            self.bank_ping = data
            
    def read_to_pe(self):
        """PE đọc dữ liệu từ bank đang hiển thị (Foreground)"""
        if self.state == "PING":
            return self.bank_ping
        else:
            return self.bank_pong
            
    def swap(self):
        """Công tắc Ping-Pong: Đảo vai trò 2 bank sau khi nạp xong"""
        self.state = "PONG" if self.state == "PING" else "PING"


class PEArray:
    """
    Mô phỏng ma trận PE (Processing Element Array).
    Được trừu tượng hóa bằng các thanh ghi ma trận Numpy để tăng tốc tính toán.
    """
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        
        # Các thanh ghi nội bộ của toàn bộ mảng PE
        self.weight_reg = None
        self.input_reg = None
        self.acc_reg = None
        
    def load_weight(self, data):
        """Khóa dữ liệu vào Weight Register (Phục vụ WS)"""
        self.weight_reg = data
        
    def load_input(self, data):
        """Khóa dữ liệu vào Input Register (Phục vụ IS)"""
        self.input_reg = data
        
    def load_psum(self, data):
        """Nạp Partial Sum từ bên ngoài vào Accumulator"""
        self.acc_reg = data
        
    def reset_accumulator(self):
        """Reset bộ cộng (Phục vụ OS)"""
        self.acc_reg = None
        
    def compute_mac(self, weight_data=None, input_data=None):
        """
        Thực thi phép nhân cộng (MAC) trên toàn bộ lưới 16x16.
        Nếu dữ liệu truyền vào là None, nó sẽ dùng dữ liệu tĩnh đang lưu trong Register.
        """
        w = weight_data if weight_data is not None else self.weight_reg
        inp = input_data if input_data is not None else self.input_reg
        
        # Mô phỏng tính toán vật lý (W * I)
        mac_result = w * inp
        
        # Spatial Reduction (Cộng dồn theo hàng)
        spatial_psum = np.sum(mac_result, axis=0)
        
        if self.acc_reg is None:
            self.acc_reg = spatial_psum
        else:
            self.acc_reg += spatial_psum
            
        return self.acc_reg
        
    def get_accumulator(self):
        """Lấy giá trị từ Accumulator Register ra ngoài (Store)"""
        return self.acc_reg


class MemoryController:
    """
    Quản lý giao tiếp DRAM/SRAM và tính toán chu kỳ pipeline (Pipeline Timing Model).
    """
    def __init__(self, config, stats):
        self.cfg = config
        self.stats = stats
        
    def fetch_weight_dram(self, size):
        self.stats.dram_weight_reads += size
        return size * self.cfg.CYCLE_DRAM_RD
        
    def fetch_input_dram(self, size):
        self.stats.dram_input_reads += size
        return size * self.cfg.CYCLE_DRAM_RD
        
    def fetch_psum_dram(self, size):
        self.stats.partial_sum_reads += size
        return size * self.cfg.CYCLE_DRAM_RD
        
    def store_psum_dram(self, size, is_final=False):
        self.stats.partial_sum_writes += size
        if is_final:
            self.stats.dram_output_writes += size
        return size * self.cfg.CYCLE_DRAM_RD
        
    def access_sram_weight(self, size):
        self.stats.sram_weight_reads += size
        return size * self.cfg.CYCLE_SRAM_RD
        
    def access_sram_input(self, size):
        self.stats.sram_input_reads += size
        return size * self.cfg.CYCLE_SRAM_RD
        
    def access_psum_pe(self, size):
        # Đọc/Ghi qua lại Register Psum
        self.stats.partial_sum_reads += size
        self.stats.partial_sum_writes += size
        
    def compute_cycles(self, num_macs):
        cycles = num_macs * self.cfg.CYCLE_MAC
        self.stats.compute_cycles += cycles
        return cycles
        
    def commit_pipeline_step(self, t_load, t_compute, t_store=0):
        """Mô phỏng độ trễ Pipeline với PingPongSRAM"""
        t_step = max(t_load, t_compute) + t_store
        self.stats.total_cycles += t_step
        self.stats.stall_cycles += max(0, t_load - t_compute)
