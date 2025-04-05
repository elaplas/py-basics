# MRO: MRO defines the order in which methods are inherited.

class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")

class C(A):
    def show(self):
        print("C")

class D(B, C):  # Inheriting from B and C
    pass

obj = D()
obj.show()
print(D.mro())  # Print MRO