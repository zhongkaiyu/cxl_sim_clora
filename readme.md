# CLoRA (CXL-based multi-LoRA serving) — SSDsim fork

This repository is the event-driven simulator for **CLoRA: A CXL-Based
System for Cost-Efficient Multi-LoRA Serving** (MICRO 2026 #785), built on
the SSDsim engine below. The C simulator models the CXL+NDP datapath; the
Python plugin (`script/clora_strategy.py`) implements the cost model and
strategy selection. Four systems are compared — **CLoRA**, **CLoRA-NoCXL**,
**Grace-Hopper**, **CPU-LoRA-Offload** — across Llama2-7B/13B (MHA),
Llama3-8B (GQA) and Qwen3-30B-A3B (GQA+MoE), on A100 and H100.

- Design & systems: [`DESIGN.md`](./DESIGN.md)
- Benchmark numbers: [`RESULTS.md`](./RESULTS.md)
- Sensitivity study: [`SENSITIVITY.md`](./SENSITIVITY.md)
- Rebuttal figures (decode TPOT, prefill TTFT): [`REBUTTAL_R2.md`](./REBUTTAL_R2.md), `fig/`

**CLoRA-NoCXL has two launch-granularity settings** (`--nocxl-fused`
selects; both are kept):
- **unfused (default, used in all plots/tables):** a kernel launch per
  serving adapter (QKV/O/FFN) and per request (attention) — CLoRA/NoCXL ≈ 4–16×.
- **fused:** S-LoRA/BGMV batching, one launch per op per layer — CLoRA/NoCXL ≈ 1.3–1.6×.

Build with `make clean && make all` (the Makefile does not track header
deps). Run the grids with `script/run_decode_all.py` / `script/run_prefill.py`
and regenerate figures with `script/make_tpot_figure.py` /
`script/make_prefill_figure.py`.

---

# SSDSim

Trace based SSD simulator.

#### Statement:

SSDsim is a simulation tool of SSDs internal hardware and software behavior. It provides specified SSDs performance, endurance and energy consumption information based on a configurable parameter file and different workloads (trace file).
SSDsim was created by Yang Hu in the end of 2009 and upgraded to version 2.0 after lots of modification and perfection. Its programming language is C and development environment is Microsoft Visual Studio 2008. With the help of Zhiming Zhu, Shuangwu Zhang, Chao Ren, Hao Luo, it is further developed into version 2.x. As the development team, we will continue adding new modules and functions to guarantee its persistent perfection. If you have any questions, suggestions or requirements about it, please feel free to email Yang Hu (yanghu@foxmail.com). We will adopt any reasonable requirements to make SSDsim better.

forked from: https://github.com/huaicheng/ssdsim

---

## List of contents:

- [How to Compile](#how-to-compile)
- [How to Run the Simulation](#how-to-run-the-simulation)
- [How to Modify parameters](#how-to-modify-parameters)
<!-- - [Run RAID Simulation](#run-raid-simulation)
- [Further Processing After Simulation](#further-processing-after-simulation)
- [Modifying the Tracefile](#modifying-the-tracefile)
  **\_\_** -->

## How to Compile

We already prepared the makefile. In unix environment, the program can be compiled by:

```
make all
```

To remove the previously compiled program, use:

```
make clean
```

If the compilation is success, there will be an executable file named `./main --file <trace_filename>`.

## How to Run the Simulation

example:

```
./main --file script/trace.json
```

quick start:

```
sh ./quick_start.sh
```

## How to Modify parameters

```
All related parameters for cxl project are listed in "config/page.paremeters"
```

## How to run the trace compare

Go to file directory, and run this python file

```
script/trace_summon.py
```

This python file will generate a random trace.json then run the program to show the Simulation Duration and Theoretical Simulation Duration.
