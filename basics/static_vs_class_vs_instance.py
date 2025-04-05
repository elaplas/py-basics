# For each variable in a class, outsite of __init__(), there are two versions of it:
# 1- class variable, which is a static variable accessed by class name and there is only one instance of it
# 2- instance variable, which is accessed by class object and there are as many of it as the number of class instances

class A:
    i = 4
    j = 5

a = A()
a.i = 2
print(A.i)
print(a.i)

class B:
    i = 4
    j = 5

    # Instance method is used in most cases and we can pass "self" to it
    def instance_method(self):
        print("I am in instance method")

    # The first parameter is always the class type and can be used to access class members inside a method class
    @classmethod 
    def func1(cls):
        x = cls.i * cls.j
        print(x)

    # There is no possiblility to access class members inside static function
    # We cannot pass "self" to it
    @staticmethod
    def func2():
       print("I am in static method")

# Both class and static methods can be called without instantiating the class as they are not associated with a class object
B.func1()
B.func2()