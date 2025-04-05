numbers = [i for i in range(20)]

res = list(map(lambda x: x*2, numbers))
print(res)

res = list(filter(lambda x: x%2==0, numbers))
print(res)

def func1(x):
    if x % 2 == 0:
        return True
    else:
        return False

res = list(filter(func1, numbers))
print(res)