import multiprocessing

def job(Q):
    print("new process is start\n")
    res = 0
    for i in range(10000000):
        res += i+i**2+i**3
    Q.put(res)  # * 将返回值放进队列'Q'
    print("new process is finished\n")

if __name__=='__main__':  # 多进程不加这句不行，多线程可以不加这句
    Q = multiprocessing.Queue()  # 创建队列
    process1 = multiprocessing.Process(target=job, args=(Q,))  # ! 函数有一个参数的时候，必须有逗号
    process2 = multiprocessing.Process(target=job, args=(Q,))

    process1.start()  # 进程1开始
    process2.start()  # 进程2开始

    process1.join()   # 进程1加入主进程
    process2.join()   # 进程2加入主进程

    res1 = Q.get()
    res2 = Q.get()
    print(res1, res2)
    print("主进程结束")