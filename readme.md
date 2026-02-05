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
