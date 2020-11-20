from numba import cuda
import numpy as np
import math
from time import time

# 確認有GPU默認使用0 : <Managed Device 0>
print(cuda.gpus)

def cpu_print(N):
    for i in range(0, N):
        print(i)

@cuda.jit
def gpu_print(N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    if (idx < N):
        print(idx)

def main():
    print("gpu print:")
    gpu_print[2, 4](8)
    cuda.synchronize()
    print("cpu print:")
    cpu_print(8)

@cuda.jit
def gpu_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        result[idx] = a[idx] + b[idx]

def mainVecAddautoMemory():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x
    gpu_result = np.zeros(n)
    cpu_result = np.zeros(n)
    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x, y, gpu_result, n)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    cpu_result = np.add(x, y)
    print("cpu vector add time " + str(time() - start))
    if (np.array_equal(cpu_result, gpu_result)):
        print("Result the same!")

def mainVecAddManualMemory():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x
    # 拷貝數據到設備端
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    # 在顯卡設備上初始化一塊用於存放GPU計算結果的空間
    gpu_result = cuda.device_array(n)
    cpu_result = np.empty(n)
    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x_device, y_device, gpu_result, n)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    cpu_result = np.add(x, y)
    print("cpu vector add time " + str(time() - start))
    if (np.array_equal(cpu_result, gpu_result.copy_to_host())):
        print("Result the same!")

if __name__ == "__main__":
    #main()
    print('Let cuda do memory allocate it self.')
    mainVecAddautoMemory()
    print('Do memory before calc.')
    mainVecAddManualMemory()
