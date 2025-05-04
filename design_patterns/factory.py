
# Factory design pattern used to manage the creation of complex objects with several variations.
# It helps hide the creation logic and enables the seperation of client and creation code.
#


from abc import abstractmethod



class Shape:
    @abstractmethod
    def draw(self):
        raise NotImplementedError

class Circle(Shape):

    def draw(self):
        print("Circle")


class Rectangle(Shape):

    def draw(self):
        print("Rectangle")

class Square(Shape):

    def draw(self):
        print("Square")

class ShapeFactory:
    @staticmethod
    def create(type:str):
        if type=="circle":
            return Circle()
        elif type == "rectangle":
            return Rectangle()
        elif type== "square":
            return Square()
        else:
            return None
        

circle = ShapeFactory.create("circle")
rec = ShapeFactory.create("rectangle")
square = ShapeFactory.create("square")
circle.draw()
rec.draw()
square.draw()