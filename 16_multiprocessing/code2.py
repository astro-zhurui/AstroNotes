import multiprocessing

def job(x):
    for i in range(10):
        x = x+x**i
    return len(str(x))

if __name__=='__main__':

    # ^ 创建进程池
    pool = multiprocessing.Pool()  # 默认使用全部CPU
    # pool = multiprocessing.Pool(processes=5)  # 指定使用CPU的核心数

    # ^ 执行运算
    # 使用刚创建的进程池pool，执行job函数的运算；
    # 函数的输入参数是列表中的元素，多核心CPU一起处理全部运算，并将结果放到results变量里
    results = pool.map(job, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) # 一共是10个不同初始值的函数运算
    print("-- job(1)的结果是：")
    print(results[0])
    print("-- job(1-10)的结果是：")
    print(results)