import prettytable as pt
from scipy.optimize import differential_evolution
import numpy as np

# 硬件参数
GPU_MEMORY = 256 * 1024**3  # GPU的显存的大小，256 GB
FLOPS_GPU = 15.7 * 10**12  # GPU计算浮点数乘法速度，15.7 TFLOPS
FLOPS_CXL = 1.57 * 10**12  # CXL计算浮点数乘法速度，1.57 TFLOPS
BAND_WIDTH = 16 * 1024**3  # GPU和CXL之间的传输速度，16 GB/s
LATENCY = 1 * 10**-6  # GPU和CXL之间的传输延迟，1微秒

# 以参数表格形式输出硬件参数
table = pt.PrettyTable()
table.field_names = ["参数", "值", "单位"]
table.add_row(["GPU_MEMORY", f"{GPU_MEMORY / 1024**3:.2f}", "GB"])
table.add_row(["FLOPS_GPU", f"{FLOPS_GPU / 10**12:.2f}", "TFLOPS"])
table.add_row(["FLOPS_CXL", f"{FLOPS_CXL / 10**12:.2f}", "TFLOPS"])
table.add_row(["BAND_WIDTH", f"{BAND_WIDTH / 1024**3:.2f}", "GB/s"])
table.add_row(["LATENCY", f"{LATENCY / 10**-6:.2f}", "us"])
print(table)

# 定义变量
Model_dim = 1024  # 模型Pretrained Weights的维度
Lora_dim = 4  # Lora微调的规模

# 定义约束条件
def constraints(vars):
    b1, b2, b3 =  vars
    constraints = []
    T_GPU_S1 = 2 * b1 * Model_dim * Lora_dim / FLOPS_GPU
    T_CXL_S1 = LATENCY + 4 * (b2 + b3) * Model_dim / BAND_WIDTH + 2 * (b2 + b3) * Model_dim * Lora_dim / FLOPS_CXL + 4 * b2 * Lora_dim / BAND_WIDTH
    constraints.append(T_CXL_S1 - T_GPU_S1)
    T_GPU_S2 = 2 * (b1 + b2) * Lora_dim * Model_dim / FLOPS_GPU
    T_CXL_S2 = 2 * b3 * Lora_dim * Model_dim / FLOPS_CXL + 4 * b3 * Model_dim / BAND_WIDTH - 4 * b2 * Lora_dim / BAND_WIDTH
    constraints.append(T_CXL_S2 - T_GPU_S2)
    constraints.append(4 * (b1 + b2) * Lora_dim + 4 * Lora_dim * Model_dim + 4 * (b1 + b2) * Model_dim - GPU_MEMORY)
    constraints.append(4 * (b1 + b2 + b3) * Model_dim - GPU_MEMORY)
    return constraints

# 定义非线性优化问题的目标函数
def objective(vars):
    b1, b2, b3 = vars
    T_GPU_S1 = 2 * b1 * Model_dim * Lora_dim / FLOPS_GPU
    T_GPU_S2 = 2 * (b1 + b2) * Lora_dim * Model_dim / FLOPS_GPU

    T_GPU = T_GPU_S1 + T_GPU_S2

    throughput = (b1 + b2 + b3) / T_GPU
    
    # 计算惩罚项
    penalty = 0
    i = 0
    # print("Constraints:", end=" ")
    for constraint in constraints(vars):
        i += 1
        if i == 3 or i == 4:
            # if constraint > 0:
            #     print(f"{i} violated: {constraint}", end=", ")
            penalty += (abs(constraint + 0.5 * 1024**3) ** 2) * 1e3  # 空闲显存超过0.5GB或者空间用超了，惩罚项为绝对值的平方
        else:
            # if constraint > 0:
            #     print(f"{i} violated: {constraint}", end=", ")
            penalty += (abs(constraint + 100 * 10**-6) ** 2) * 1e6  # CXL空闲时间超过100us或者CXL时间超过GPU时间，惩罚项为绝对值的平方
    # print()
    
    return -throughput + penalty  # maximize throughput by minimizing negative throughput and adding penalty for constraints

# 初始猜测值
initial_guess = [16 * 1024**2, 128, 1024 * 2]  # 假设b1, b2, b3的初始值

# 边界条件
bounds = [(2 * 1024**2, 1024**3), (0, 1024), (1, 256**2)]  # 假设b1, b2, b3的上限

# 使用SciPy的differential_evolution函数求解非线性优化问题
result = differential_evolution(objective, bounds=bounds, strategy='best1bin', maxiter=100000, popsize=100, tol=0.01)

# 将结果取整
result.x = np.round(result.x).astype(int)

# 输出结果
print("Status:", result.success)
print("b1:", result.x[0])
print("b2:", result.x[1])
print("b3:", result.x[2])
print("Throughput:", (result.x[0] + result.x[1] + result.x[2]) / (2 * result.x[0] * Model_dim * Lora_dim / FLOPS_GPU + 2 * (result.x[0] + result.x[1]) * Lora_dim * Model_dim / FLOPS_GPU))

# 打印详细运算过程
b1, b2, b3 = result.x
T_GPU_S1 = 2 * b1 * Model_dim * Lora_dim / FLOPS_GPU
T_CXL_S1 = LATENCY + 4 * (b2 + b3) * Model_dim / BAND_WIDTH + 2 * (b2 + b3) * Model_dim * Lora_dim / FLOPS_CXL + 4 * b2 * Lora_dim / BAND_WIDTH
T_GPU_S2 = 2 * (b1 + b2) * Lora_dim * Model_dim / FLOPS_GPU
T_CXL_S2 = 2 * b3 * Lora_dim * Model_dim / FLOPS_CXL + 4 * b3 * Model_dim / BAND_WIDTH - 4 * b2 * Lora_dim / BAND_WIDTH
throughput = (b1 + b2 + b3) / (max(T_GPU_S1, T_CXL_S1) + max(T_GPU_S2, T_CXL_S2))
Min_GPU_MEMORY = min(GPU_MEMORY - 4 * (b1 + b2 + b3) * Model_dim, GPU_MEMORY - 4 * (b1 + b2) * Lora_dim - 4 * Lora_dim * Model_dim - 4 * (b1 + b2) * Model_dim)
print()
print("详细运算过程:")
print("第一阶段 (Stage 1):")
print("  GPU 计算耗时 T_GPU_S1:", f"{T_GPU_S1 * 10**6:.2f}", "us")
print("  CXL 计算耗时 T_CXL_S1:", f"{T_CXL_S1 * 10**6:.2f}", "us")
print("第二阶段 (Stage 2):")
print("  GPU 计算耗时 T_GPU_S2:", f"{T_GPU_S2 * 10**6:.2f}", "us")
print("  CXL 计算耗时 T_CXL_S2:", f"{T_CXL_S2 * 10**6:.2f}", "us")
print("总计算耗时 (Total Time):", f"{(max(T_GPU_S1, T_CXL_S1) + max(T_GPU_S2, T_CXL_S2)) * 10**6:.2f}", "us")
print("运算中最小GPU内存剩余 (Minimum GPU Memory Remaining):", f"{Min_GPU_MEMORY / 1024**3:.2f}", "GB")
print("最终吞吐量 (Throughput):", throughput, "ops/s")