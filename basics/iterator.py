#Use iter() function to create an iterator
l = [1,2,3,4]
it1 = iter(l)

for i in range(len(l)):
    # Use next() to fetch/retrieve the next element
    x = next(it1)
    print(x)

print("..................")

# Implement __iter__() and __next__() to make a type iterator
# The state of the next value is manually mantained in __next__()
class A:
    # Always return "self"
    s = 0
    e = 20
    def __iter__(self):
        return self

    def __next__(self):
        if self.s > self.e:
            raise StopIteration
        tmp = self.s
        self.s +=2 
        return tmp

a1 = A()
for i in range(10):
    x = next(a1)
    print(x)

print("..................")
a2 = A()
for x in a2:
    print(x)

# Implement __iter__() using yield
# Hint: such iterator cannot be used by next()
class B:
    def __iter__(self):
        for i in range(20):
            if i % 2 == 0:
                yield i

print("..................")
b2 = B()
for x in b2:
    print(x)

#Gives error
print("..................")
b1 = B()
for i in range(10):
    x = next(b1)
    print(x)
