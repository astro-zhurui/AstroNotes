import multiprocessing
import os

def job(x):
    print(f"x={x}, pid={os.getpid()}")
    for i in range(10):
        x = x+x**i
    return len(str(x))

print(f"if语句之前, pid={os.getpid()}")


print(f"进入if语句, pid={os.getpid()}")
# ^ 创建进程池
pool = multiprocessing.Pool(processes=3)  # 默认使用全部CPU
# ^ 执行运算
results = pool.map(job, (1, 2, 3, 4, 5)) # 一共是10个不同初始值的函数运算
# ^  关闭进程池
pool.close()
print(f"程序结束, pid={os.getpid()}")