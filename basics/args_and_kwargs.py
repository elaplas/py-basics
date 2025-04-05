
def func1(*args):
    for arg in args:
        print(arg)

def func2(**kwargs):
    for k in kwargs:
        print(k)
        print(kwargs[k])


def func3(*args, **kwargs):
    for arg in args:
        print(arg)
    
    for k in kwargs:
        print(k)
        v = kwargs[k]
        print(v)

func1(1,2,4.5, "ooo", [10, 20, 30] )
print(".................")
func2(name="Ebi", age=39)
print(".................")
func3(7,8,9, name="Ebi")