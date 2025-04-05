class A:
    def func1(self):
        print("I am A")

class B:
    def func2(self):
        print("I am B")

class C(A,B):
    def func3(self):
        print("I am C")
        
#Child class inherits from both Parent1 and Parent2.
#It can use methods from both parent classes.
c = C()
c.func1()
c.func2()
c.func3()