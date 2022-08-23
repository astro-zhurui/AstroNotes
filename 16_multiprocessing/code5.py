import multiprocessing
import time

lock = multiprocessing.Lock()  # ! 必须写在主函数中

def job(v, num, process_name, lock):   # ! 注意这里添加个lock
    lock.acquire()  # * 获取进程锁
    for _ in range(10):
        time.sleep(0.5)
        v.value += num
        print("{}: {}".format(process_name, v.value))
    lock.release()  # * 释放进程锁

if __name__=='__main__':

    print("--- 演示使用进程锁, 防止多进程争抢共享内存里的变量v")
    v = multiprocessing.Value('i', 0) # 创建共享内存里的变量

    process1 = multiprocessing.Process(target=job, args=(v, 1, 'process 1', lock))
    process2 = multiprocessing.Process(target=job, args=(v, 100, 'process 2', lock))
    process1.start()
    process2.start()
    process1.join()
    process2.join()