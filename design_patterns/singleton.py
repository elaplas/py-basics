
# Singleton design pattern is used to: 
# - Coordinate system-wide actions for example
#   - Ensuring there is only one instance of a type used in different parts of an applications to access 
#     a database to write it. If there would be many instances of this type, it could lead to undefined modifications
#     of the database
#   - Ensuring there is only one instance of a type used in different parts of an applications for logging 
#     in the same place and with the same format
#   - Ensuring there is only one instance of a type used in different parts of an applications for writing 
#     and reading a catche memory 
#   
# - Providing system-wide configurations and settings for consistency and efficiency
#
#

class Configer:
    _configer = None

    def __init__(self):
        self.data = None
    
    def setConfig(self, val):
        self.data = val
    
    def getConfig(self):
        return self.data
    
    def __new__(cls):
        if cls._configer is None:
            cls._configer = super().__new__(cls)
        return cls._configer

configer1 = Configer()
configer2 = Configer()

print(configer1 is configer2)