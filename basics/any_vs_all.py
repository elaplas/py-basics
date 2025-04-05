
# any: if at least one is true 
# all: all must be true

print(any([True, False, False]))
print(all([True, True, True]))
print(all([True, True, False]))

l1 = [1,2,3,4,5]
l2 = [1,2,3,4,-5]

if any(el < 0 for el in l2 ):
    print("yes")

if all(el > 0 for el in l1 ):
    print("yes")

if any(el < 0 for el in l1 ):
    print("yes")