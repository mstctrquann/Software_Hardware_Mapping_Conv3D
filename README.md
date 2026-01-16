# 🚀 Cycle-Accurate CNN Accelerator Simulator

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![Verilog](https://img.shields.io/badge/Hardware-Verilog-red.svg) ![Status](https://img.shields.io/badge/Status-Research-orange.svg)

## 📖 Overview
This project is a **cycle-accurate behavioral model** of a specialized hardware accelerator for Convolutional Neural Networks (CNNs). Developed using Python/Numpy, the simulator enables **Design Space Exploration (DSE)** to analyze trade-offs between compute performance, memory bandwidth, and on-chip SRAM area.

The architecture implements a **Weight Stationary** dataflow optimized for Edge AI devices, addressing the "Memory Wall" bottleneck through advanced buffering techniques like **Line Buffers** and **Ping-Pong SRAM** under strict **Single-Port DRAM** constraints.

## 🏗️ System Architecture

### Block Diagram
*(Recommend: Insert your system block diagram here to visualize the interconnects between DRAM, SRAM, and the PE Array)*

### Data Flow (Data Path)
The architecture optimizes data movement through the following simplified paths:
* **Weight Path:** DRAM → Weight Bank (Load) → Swap → Weight Bank (Compute) → PE Array.
* **Input Path:** DRAM → Line Buffer → Input Bank → PE Array (Broadcast).
* **Output Path:** PE Array → Adder Tree → Global Accumulator → DRAM.

## ✨ Key Features
* **Cycle-Accurate Timing:** Precise modeling of clock cycles for DRAM accesses, SRAM R/W operations, and MAC execution.
* **Weight Stationary Dataflow:** Minimizes weight movement energy by keeping weights static within PEs during input streaming.
* **Latency Hiding:** Implements **Ping-Pong Buffering** to overlap DRAM access time with active computation cycles.
* **Bandwidth Optimization:** Integrated **Line Buffer** logic to significantly reduce DRAM traffic by maximizing vertical spatial reuse of input feature maps.

## 🛠️ Hardware Implementation (RTL)
The project includes Register-Transfer Level (RTL) designs for the core computational blocks:
* **Processing Element (PE):** A 32-bit signed integer MAC unit featuring an internal accumulator and control logic (enable/clear).
* **Adder Tree:** A 4-stage binary adder tree designed to reduce results from 16 PE rows into a single output stream.


## 📂 Project Structure
```text
├── hardware_accel_v3.py    # Main Simulator: PE Array, SRAM, and Dataflow logic
├── processing_element.v    # RTL design for the core PE computational unit
├── benchmark_data.npz      # Verified sample Input/Weight tensors for consistency
└── README.md               # Project documentation
