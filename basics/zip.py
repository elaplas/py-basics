## Combine as tuples and returns it as iterator
values = [20, 25, 21, 23, 24, 25]
names = ['a', 'b', 'c', 'd', 'e', 'f']
z = list(zip(names, values))
print(z)

for x in zip(names, values):
    print(x)
