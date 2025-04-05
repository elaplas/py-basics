from multiprocessing import Pool

def func(data):
    res = 0
    for el in data:
        res += el
    return res

data = [[i for i in range(10)] for _ in range(5)]

num_workers = 5
with Pool(num_workers) as pool:
    res = pool.map(func, data)
print(res)

