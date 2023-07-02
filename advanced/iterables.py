

################# Generators ################ 
# "yield" makes a function an iterable. 
# "yield from" makes a function with yield inside another function an iterable 
# Example: 

def func(n):
    if n==0:
        return
    
    if n%2==0:
        yield n

    yield from func(n-1)

def even(n):
    for i in range(n):
        if i%2 ==0:
            yield i

def odd(n):
    for i in range(n):
        if i%2 != 0:
            yield i

def real(n):
    yield from odd(n)
    yield from even(n)


evens = [i for i in func(10)]
reals = [i for i in real(10)]
print(evens)
print(reals)

################### zip #################
# Makes tuples from lists

odd_list = [i for i in range(10) if i%2!=0]
even_list = [i for i in range(10) if i%2==0]
real_list = [i for i in range(10)]

tuples = zip(even_list, odd_list, real_list)
print(*tuples)

################# dicts ##################
# Generate dicts inside another dict
maps = {i: {} for i in range(10)}
print(maps)

    
