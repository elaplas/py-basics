# Decorator is a wrapper that takes a function, adds more functionality
# to it without modifying it and then returns it

# Use case: keep core logic seperated from logging and debugging logic
# - Logging and debugging 
# - Extending function without modifying it
# - Reusability


import time


##Logging use case
def time_decorator(function):
    def wrapper(*args, **kws):
        start = time.perf_counter()
        res = function(*args, **kws)
        print(f"{function.__name__} took {time.perf_counter()-start} s")
        return res
    return wrapper

@time_decorator
def heavy_operation():
    res = 0
    for i in range(9000000):
        res += i
    return res

res = heavy_operation()
print(f"heavy operation restult: {res}")

# Chaining decorators
# They are executed from bottom up

## Use case: making a text first italic then bold
def bold(function):
    def wrapper(*args):
        return "<b>" + function(*args) + "</b>"
    return wrapper

def italic(function):
    def wrapper(*args):
        return "<i>" + function(*args) + "</i>"
    return wrapper

@bold
@italic
def mytext():
    return "I am Ebi"

print(mytext())