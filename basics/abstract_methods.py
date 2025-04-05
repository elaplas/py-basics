
from abc import abstractmethod

class A:
    @abstractmethod
    def sound(self):
        pass

class B(A):
    def sound(self):
        print("BBB")


class C(A):
    def sound(self):
        print("CCC")


def func(a: A):
    a.sound()

b = B()
c = C()
b.sound()
#c.sound()
func(c)



#Short version
#
#ABCs offer a higher level of semantic contract between clients and the implemented classes.
#Long version
#
#There is a contract between a class and its callers. The class promises to do certain things and have certain properties.
#
#There are different levels to the contract.
#
#At a very low level, the contract might include the name of a method or its number of parameters.
#
#In a staticly-typed language, that contract would actually be enforced by the compiler. 
# Python, you can use EAFP or type introspection to confirm that the unknown object meets this expected contract.
#
#But there are also higher-level, semantic promises in the contract.
#
#For example, if there is a __str__() method, it is expected to return a string representation of the object. 
# It could delete all contents of the object, commit the transaction and spit a blank page out of the printer... but there is a common understanding of what it should do, described in the Python manual.
#
#That's a special case, where the semantic contract is described in the manual. What should the print() method do? 
# Should it write the object to a printer or a line to the screen, or something else? It depends - you need to read the comments to understand the full contract here. A piece of client code that simply checks that the print() method exists has confirmed part of the contract - that a method call can be made, but not that there is agreement on the higher level semantics of the call.
#
#Defining an Abstract Base Class (ABC) is a way of producing a contract between the class implementers and the callers. 
# It isn't just a list of method names, but a shared understanding of what those methods should do. If you inherit from this ABC, you are promising to follow all the rules described in the comments, including the semantics of the print() method.
#
#Python's duck-typing has many advantages in flexibility over static-typing, but it doesn't solve all the problems.
# ABCs offer an intermediate solution between the free-form of Python and the bondage-and-discipline of a staticly-typed language.
#