import sys

# How does Python manage memory?
# Python manages memory using reference counting and garbage collection.
# Python uses reference counting to track objects.
# When an object’s reference count reaches zero, it gets garbage collected.

x = [2, 1, 3]

def func():
    y = x
    print(f"reference cout for x  inside fun(): ", sys.getrefcount(x)) # Output: 3 (including sys.getrefcount itself)
    print(f"reference cout for y  inside fun(): ", sys.getrefcount(y)) # Output: 3 (including sys.getrefcount itself)

func()
print(f"reference cout for x  inside fun(): ", sys.getrefcount(x))