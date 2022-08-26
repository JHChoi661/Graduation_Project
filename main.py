from concurrent.futures import process
import subprocess
from multiprocessing import Process, Queue
from time import sleep
def sub1(q):
    proc = subprocess.Popen('C:\gradProject\Source\HandGestureDetection.py', shell=True, stdout=subprocess.PIPE, encoding='utf-8')
    while True:
        out = proc.stdout.readline()
        if out == '' and proc.poll() is not None:
            break
        if out:
            q.put(out.strip())


if __name__ == '__main__':


    ### 손 제스처 인식 main
    q = Queue()
    p = Process(target=sub1, args=(q,))
    p.start()
    while True:
        res = q.get()
        """
        라즈베리파이에 명령 전송
        (1개씩 바로 간다. 라즈베리파이에서도 동시에 처리
        -> 블루투스 기기면 페어링, IR 기기면 서버에 테이블 요청?)
        """
        print(res)
        if res == 'Down':
            p.kill()
            break
        
        
    p.join()

