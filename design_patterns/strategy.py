
#Strategy Pattern
#Purpose: Allows selecting an algorithm’s behavior at runtime.
#
#Use case: Sorting algorithms, different payment methods in e-commerce, 
#or AI decision trees.
#
#Benefit: Makes algorithms interchangeable and avoids hardcoding logic.


class CompressionStrategy:
    def compress(self):
        raise NotImplementedError

class Compressor:
    def __init__(self, compressor: CompressionStrategy):
        self.compressor = compressor
    def compress(self):
        self.compressor.compress()

class RarCompressor(CompressionStrategy):
    def compress(self):
        print("compressing by Rar")

class ZipCompressor(CompressionStrategy):
    def compress(self):
        print("compressing by Zip")

compressor1 = Compressor(RarCompressor())
compressor2 = Compressor(ZipCompressor())
compressor1.compress()
compressor2.compress()