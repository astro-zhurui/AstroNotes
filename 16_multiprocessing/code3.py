import multiprocessing

def job(x):
    for i in range(10):
        x = x+x**i
    return len(str(x))

if __name__=='__main__':

    # ^ 创建进程池
    pool = multiprocessing.Pool()  # 默认使用全部CPU
    # pool = multiprocessing.Pool(processes=5)  # 指定使用CPU的核心数
    pool.close()  # 执行完得关闭，不然会报错

    # ^ 执行运算
    # 使用刚创建的进程池pool，执行job函数的运算；
    # 只能输入一个任务的参数，返回一个任务的结果
    result = pool.apply_async(job, (1,))
    res = result.get()  # 使用get方法获得返回值
    print("job(1)的结果是: ")
    print(res)

    # ^ 如果想使用apply_async实现map的效果，需要对此方法迭代
    results = [pool.apply_async(job, (i,)) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    list_res = [res.get() for res in results]
    print("job(1-10)的结果是: ")
    print(list_res)
