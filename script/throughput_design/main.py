import prettytable as pt
import pulp

"本代码中所有的时间参数的单位都是秒(s),所有的数据大小参数的单位都是字节(B)"
# 硬件参数
GPU_MEMORY = 80 * 10**9  # GPU的显存的大小，128 GB FIXME：80GB用于存储模型参数，48GB用于存储中间结果
FLOPS_GPU = 312 * 10**12  # GPU计算浮点数乘法速度，312 TFLOPS 「Tera Floating Point Operations Per Second」
GPU_EFFICIENCY = 0.8 # GPU能发挥的算力大小
FLOPS_CXL = 4 * 10**12  # CXL计算浮点数乘法速度，4 TFLOPS
CXL_EFFICIENCY = 0.8 # CXL能发挥的算力大小
BAND_WIDTH = 128 * 10**9  # GPU和CXL之间的传输速度，128 GB/s
LATENCY = 0.2 * 10**-6  # GPU和CXL之间的传输延迟，0.2微秒

# 默认参数
FLOAT_SIZE = 2  # 一个Float16占2B
BATCH_SIZE = 32  # Batch size（num_GPU_batches）GPU默认处理的任务数
MODEL_DIM = 8192    # 模型Pretrained Weights（d*d的矩阵）的维度d 8*1024
LORA_DIM = 4       # Lora微调的（A和B都是d*r的矩阵）的规模r，设置为4，r<<d

# 以参数表格形式输出硬件参数和默认参数
table = pt.PrettyTable()
table.field_names = ["参数", "值", "单位"]
table.add_row(["GPU_MEMORY", f"{GPU_MEMORY / 10**9:.2f}", "GB"])
table.add_row(["FLOPS_GPU", f"{FLOPS_GPU / 10**12:.2f}", "TFLOPS"])
table.add_row(["GPU_EFFICIENCY", f"{GPU_EFFICIENCY / 10**-2:.2f}", "%"])
table.add_row(["FLOPS_CXL", f"{FLOPS_CXL / 10**12:.2f}", "TFLOPS"])
table.add_row(["CXL_EFFICIENCY", f"{CXL_EFFICIENCY / 10**-2:.2f}", "%"])
table.add_row(["BAND_WIDTH", f"{BAND_WIDTH / 10**9:.2f}", "GB/s"])
table.add_row(["LATENCY", f"{LATENCY / 10**-6:.2f}", "us"])
table.add_row(["FLOAT_SIZE", f"{FLOAT_SIZE}", "B"])
table.add_row(["BATCH_SIZE", f"{BATCH_SIZE}", ""])
table.add_row(["MODEL_DIM", f"{MODEL_DIM}", ""])
table.add_row(["LORA_DIM", f"{LORA_DIM}", ""])
print(table)

# 定义线性规划问题
prob = pulp.LpProblem("Memory_Optimization", pulp.LpMaximize)
# 定义正整数变量 BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3 (则GPU一次性处理Z=XAB类型时的任务数为 BLOCK_SIZE * BATCH_SIZE)
# blk1 = pulp.LpVariable('blk1', lowBound=0, upBound=1, cat='Integer')
# blk2 = pulp.LpVariable('blk2', lowBound=0, upBound=1, cat='Integer')
blk1 = 0
blk2 = 0
blk3 = pulp.LpVariable('blk3', lowBound=0, upBound=1, cat='Integer')
# 约束条件：最开始对于GPU的内存使用不超过GPU_MEMORY
prob += FLOAT_SIZE * (blk1 + blk2 + blk3) * BATCH_SIZE * MODEL_DIM <= GPU_MEMORY # blk1、blk2、blk3的内存使用


"第一阶段(Stage1),计算Y = XA,GPU把 batch2 和 batch3 的部分的矩阵卸载到CXL中计算,在GPU中计算 batch1 的部分,CXL计算并将 batch2 结果传回GPU,传回时刻对齐,对结果进行整合"
# GPU第一阶段耗时为：计算Y = XA
T_GPU_S1 = 2 * blk1 * MODEL_DIM * LORA_DIM / FLOPS_GPU
# CXL第一阶段耗时为：指令延迟 + 传输b2+b3数据 + 计算Y = XA + 传输b2数据
T_CXL_S1 = LATENCY + FLOAT_SIZE * (blk2 + blk3) * MODEL_DIM / BAND_WIDTH + 2 * (blk2 + blk3) * MODEL_DIM * LORA_DIM / FLOPS_CXL  + FLOAT_SIZE * blk2 * LORA_DIM / BAND_WIDTH
# 约束条件：CXL第一阶段的时间不超过GPU第一阶段的时间，能够赶上第一班“车”
prob += T_CXL_S1 <= T_GPU_S1
# 约束条件：第一阶段计算过程中和计算后拼接b1和b2，对于GPU的内存使用不超过GPU_MEMORY
prob += FLOAT_SIZE * blk1 * MODEL_DIM + FLOAT_SIZE * MODEL_DIM * LORA_DIM + FLOAT_SIZE * blk1 * LORA_DIM <= GPU_MEMORY # X(b1)、A、Y(b1)的内存使用
prob += FLOAT_SIZE * (blk1 + blk2) * LORA_DIM <= GPU_MEMORY # b1、b2的内存使用
# 第一阶段总耗时
print("T_GPU_S1:", T_GPU_S1, "T_CXL_S1:", T_CXL_S1)
# 第一阶段总内存使用
print("M_GPU_S1-1:", FLOAT_SIZE * blk1 * MODEL_DIM + FLOAT_SIZE * MODEL_DIM * LORA_DIM + FLOAT_SIZE * blk1 * LORA_DIM, "M_GPU_S1-2:", FLOAT_SIZE * (blk1 + blk2) * LORA_DIM)


"第二阶段(Stage2),计算Z = YB,在GPU中计算 batch1 和 batch2 的部分,CXL计算并将 batch3 结果传回GPU,传回时刻对齐,对结果进行整合"
# GPU第二阶段耗时为：计算Z = YB
T_GPU_S2 = 2 * (blk1 + blk2) * LORA_DIM * MODEL_DIM / FLOPS_GPU
# CXL第二阶段耗时为：计算Z = YB + 传输b3数据 - 传输b2数据 TODO: 传输b3是否需要指令，是否需要考虑LATENCY？
T_CXL_S2 = 2 * blk3 * LORA_DIM * MODEL_DIM / FLOPS_CXL + FLOAT_SIZE * blk3 * MODEL_DIM / BAND_WIDTH - FLOAT_SIZE * blk2 * LORA_DIM / BAND_WIDTH
# 约束条件：CXL第二阶段的时间不超过GPU第二阶段的时间，能够赶上第二班“车”
prob += T_CXL_S2 <= T_GPU_S2
# 约束条件：第二阶段计算过程中和计算后拼接b1b2和b3，对于GPU的内存使用不超过GPU_MEMORY
prob += FLOAT_SIZE * (blk1 + blk2) * LORA_DIM + FLOAT_SIZE * LORA_DIM * MODEL_DIM + FLOAT_SIZE * (blk1 + blk2) * MODEL_DIM <= GPU_MEMORY # Y(b1b2)、B、Z(b1b2)的内存使用
prob += FLOAT_SIZE * (blk1 + blk2 + blk3) * MODEL_DIM <= GPU_MEMORY # b1、b2、b3的内存使用
# 第二阶段总耗时
print("T_GPU_S2:", T_GPU_S2, "T_CXL_S2:", T_CXL_S2)
# 第二阶段总内存使用
print("M_GPU_S2-1:", FLOAT_SIZE * (blk1 + blk2) * LORA_DIM + FLOAT_SIZE * LORA_DIM * MODEL_DIM + FLOAT_SIZE * (blk1 + blk2) * MODEL_DIM, "M_GPU_S2-2:", FLOAT_SIZE * (blk1 + blk2 + blk3) * MODEL_DIM)


# 目标函数：最大化吞吐率
prob += (blk1 + blk2 + blk3) / (T_GPU_S1+ T_GPU_S2)
# 输出线性规划问题
print(prob)
# 求解问题
prob.solve()
# 输出结果
print("Status:", pulp.LpStatus[prob.status])
print("b1:", pulp.value(blk1))
print("b2:", pulp.value(blk2))
print("b3:", pulp.value(blk3))
print("Throughput:", pulp.value(prob.objective))