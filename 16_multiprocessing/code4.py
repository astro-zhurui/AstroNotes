import multiprocessing
import time

def job(v, num, process_name):
    for _ in range(10):
        time.sleep(0.5)
        v.value += num
        print("{}: {}".format(process_name, v.value))

if __name__=='__main__':
    print("--- 演示多进程在争抢共享内存里的变量v")
    v = multiprocessing.Value('i', 0)
    process1 = multiprocessing.Process(target=job, args=(v, 1, 'process 1'))
    process2 = multiprocessing.Process(target=job, args=(v, 100, 'process 2'))
    process1.start()
    process2.start()
    process1.join()
    process2.join()