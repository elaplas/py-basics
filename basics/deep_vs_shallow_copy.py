import copy



original = [[1, 2, 3], [4, 5, 6]]

# Creates a new object (of the outer) but does not recursively copy objects inside it. Changes in nested objects affect both copies.
shallow_copy = copy.copy(original)  	
# Recursively copies all objects inside the original, creating a completely independent copy.
deep_copy = copy.deepcopy(original)

original[0][0] = 99

print(shallow_copy)
print(deep_copy)