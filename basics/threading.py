import threading

counter = 0

def increment():
    for _ in range(1000000):
        counter += 1

thread1 = threading.Thread(target = increment)
thread2 = threading.Thread(target = increment)

thread1.start()
thread2.start()
thread1.join()
thread2.join()

print(counter)