# Observer design patterns enables loose coupling and it is used to allow one object (subject)
# notify other objects (observers) about changes in its state. 
#
# Loose Coupling:
#   The subject and observers don't need to know about each other's implementations,
#   making the system more modular and easier to.
# Flexibility and Extensibility:
#   Adding or removing observers doesn't require modifying the subject, making the system easier to extend.
# Event Handling:
#  The pattern is well-suited for scenarios where you need to notify multiple objects about events 
#  or changes in the state of another object.
# 
# 
# Example:
# Consider a weather station (Subject) that collects temperature, humidity, and pressure. Multiple displays
# (Observers) need to display this data. Without the Observer pattern, each display would need to poll the 
# weather station constantly for updates. With the pattern, the weather station notifies the displays whenever 
# the data changes, and each display only receives the update.
#

from abc import abstractmethod

class Observer:
    @abstractmethod
    def update(self, t: float):
        raise NotImplementedError

class Subject:
    @abstractmethod
    def addObserver(self, observer:Observer):
        raise NotImplementedError
    @abstractmethod
    def removeObserver(self, observer:Observer):
        raise NotImplementedError
    @abstractmethod
    def notifyObservers(self):
        raise NotImplementedError
    
class Display(Observer):
    def __init__(self, id:int):
        self.id = id
        self.t = None
    def update(self, t: float):
        self.t = t
    def show(self):
        print(f"display {self.id} : temprature: {self.t} °C")

class WeatherStation(Subject):
    def __init__(self):
        self.observers = set()
        self.t = None

    def setTemprature(self, t):
        self.t = t
    
    def addObserver(self, observer:Observer):
        self.observers.add(observer)
    
    def removeObserver(self, observer:Observer):
        self.observers.remove(observer)

    def notifyObservers(self):
        for observer in self.observers:
            observer.update(self.t)



display1 = Display(1)
display2 = Display(2)
display3 = Display(3)

weatherStation = WeatherStation()
weatherStation.setTemprature(23.3)
weatherStation.addObserver(display1)
weatherStation.addObserver(display2)
weatherStation.addObserver(display3)
weatherStation.notifyObservers()

display1.show()
display2.show()
display3.show()

    

