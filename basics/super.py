# super() is used to call methods from a parent class
class A:
    def show(self):
        print("Parent method")

class B(A):
    def show(self):
        super().show()
        print("Child method")


b = B()
b.show()