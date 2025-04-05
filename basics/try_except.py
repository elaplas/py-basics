# try - catch in python
def func1(x, y):
    try:
        x / y
    except ZeroDivisionError:
        print("zero division error")
    except TypeError:
        print("type error")
    except:
        print("other errors")

func1(2, 0)
func1(2, "aaa")

# throw an exception in python
def func2(x):
    if type(x) != list:
        raise Exception("input should be a list")
    for i in range(len(x)):
        print(x[i])

func2(3)
