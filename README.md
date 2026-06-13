# Conv3D Accelerator Dataflow Exploration

## 1. Overview

This project models the computation hardware of a Conv3D layer and evaluates the impact of different dataflow strategies on memory access patterns and execution performance.

The objective is to compare the following dataflow strategies under the same hardware architecture and workload configuration:
- **Baseline** (No Stationary)
- **Weight Stationary** (WS)
- **Input Stationary** (IS)
- **Output Stationary** (OS)

---

## 2. Workload Configuration

**Dimensions & Kernel:**
- `D = 16`, `H = 32`, `W = 32`
- `C_in = 64`
- `K_D = 3`, `K_H = 3`, `K_W = 3`

**Output Dimensions:**
- `D_out` = D - K_D + 1 = **14**
- `H_out` = H - K_H + 1 = **30**
- `W_out` = W - K_W + 1 = **30**

---

## 3. Hardware Architecture

### Processing Element (PE) Array
- `PE_ROWS = 16`
- `PE_COLS = 16`

**PE Internal Structure:**
```text
PE
├── Weight Register
├── Input Register
├── Accumulator Register
└── MAC Unit
```

### Memory Hierarchy
```text
DRAM
 ├── Weight Buffer (Implemented as PingPongSRAM)
 ├── Input Buffer (Implemented as PingPongSRAM)
 └── Output Buffer
```

**Ping-Pong Buffering (Double Buffering):**
To optimize execution cycles and hide DRAM latency, both the Weight Buffer and Input Buffer are physically implemented using `PingPongSRAM`. This double-buffering architecture allows the PE array to continuously compute data from `SRAM_0` while new data is pre-fetched from DRAM into `SRAM_1` (and vice versa).

### Data Path Overview

The physical architecture remains unchanged for all experiments; only the dataflow policy is modified.

- **Weight Path:** `DRAM → Weight PingPongSRAM (Prefetch) → PE Array (Compute)`
- **Input Path:** `DRAM → Input PingPongSRAM (Prefetch) → PE Array (Compute)`
- **Output Path:** `PE Array → Adder Tree → Output Buffer → DRAM`

---

## 4. Dataflow Strategies

### 4.1 Baseline Dataflow (No Stationary)
**Motivation:** Serves as a neutral reference point. No data type is intentionally kept stationary. All operands are continuously streamed from SRAM buffers.

- **Weight:** `DRAM → Weight Buffer → PE` (Fetched whenever needed; no explicit reuse inside PE)
- **Input:** `DRAM → Input Buffer → PE` (Streamed continuously; no explicit reuse inside PE)
- **Output:** `PE → Adder Tree → Output Buffer → DRAM` (Partial sums are immediately reduced and written back)

**Characteristics:**
- ✅ **Advantages:** Simplest implementation, minimal local storage.
- ❌ **Disadvantages:** High memory traffic, poor data reuse.

### 4.2 Weight Stationary (WS)
**Principle:** Weights remain inside PE registers as long as possible. Input activations are streamed. Outputs are accumulated normally.

- **Weight:** `DRAM → Weight Buffer → PE Weight Register` (Loaded once, reused multiple times)
- **Input:** `DRAM → Input Buffer → PE` (Continuously streamed)
- **Output:** `PE → Adder Tree → Output Buffer → DRAM`

**Expected Benefit:**
- ⬇️ **Reduce:** Weight DRAM Access, Weight SRAM Access
- ⬆️ **Increase:** Weight Reuse Factor

### 4.3 Input Stationary (IS)
**Principle:** Input activations remain inside PE registers as long as possible. Weights are streamed through the array.

- **Input:** `DRAM → Input Buffer → PE Input Register` (Loaded once and reused)
- **Weight:** `DRAM → Weight Buffer → PE` (Streamed continuously)
- **Output:** `PE → Adder Tree → Output Buffer → DRAM`

**Expected Benefit:**
- ⬇️ **Reduce:** Input DRAM Access, Input SRAM Access
- ⬆️ **Increase:** Input Reuse Factor

### 4.4 Output Stationary (OS)
**Principle:** Partial sums remain inside the PE accumulator until the final output value is completed. This strategy minimizes intermediate output movement.

- **Weight:** `DRAM → Weight Buffer → PE`
- **Input:** `DRAM → Input Buffer → PE`
- **Output:** `PE Accumulator → Output Buffer → DRAM` (Partial sums never leave the PE during computation; only final outputs are written back)

**Expected Benefit:**
- ⬇️ **Reduce:** Partial Sum SRAM Writes/Reads, Output Traffic
- ⬆️ **Increase:** Accumulator Utilization

---

## 5. Evaluation Metrics

To ensure a fair comparison, all experiments use the same physical parameters. Only the dataflow policy changes.

**Hardware & Workload:**
```python
PE_ROWS, PE_COLS = 16, 16
Clock = 200 MHz
D, H, W = 16, 32, 32
C_in = 64
Kernel = 3x3x3
```

### Metrics Collected

| Category | Metrics |
| :--- | :--- |
| **Compute** | `total_mac`, `compute_cycles`, `pe_utilization`, `throughput_mac_per_cycle` |
| **Load** | `dram_weight_reads`, `dram_input_reads`, `sram_weight_reads`, `sram_input_reads` |
| **Store** | `dram_output_writes`, `partial_sum_reads`, `partial_sum_writes` |
| **Reuse** | `weight_reuse_factor`, `input_reuse_factor`, `output_reuse_factor` |

---

## 6. Fair Comparison Methodology

The following parameters **MUST** remain identical across all tests:
- PE array size
- Clock frequency
- DRAM bandwidth
- SRAM capacity
- Conv3D workload dimensions

Only the mapping policy changes (`Dataflow.BASELINE`, `Dataflow.WS`, `Dataflow.IS`, `Dataflow.OS`). This ensures that any observed performance difference is caused solely by the dataflow strategy.

---

## 7. Expected Outcome

| Dataflow | Primary Reuse Target | Reduced Traffic |
| :--- | :--- | :--- |
| **Baseline** | None | None |
| **WS** | Weight | Weight Access |
| **IS** | Input Activation | Input Access |
| **OS** | Partial Sum | Output / Psum Access |

The study aims to quantify how each dataflow affects memory traffic, hardware utilization, and execution cycles for Conv3D workloads.
