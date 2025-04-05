# List - Ordered, mutable
my_list = [1, 2, 3]
my_list.append(4)
print(my_list)  # [1, 2, 3, 4]

# Tuple - Ordered, immutable
my_tuple = (1, 2, 3)
print(my_tuple[1])  # 2

# Set - Unordered, unique elements
my_set1 = {1, 2, 3, 3, 2, 1}
my_set2 = set([1, 2, 3, 3, 2, 1])
print(my_set1)  # {1, 2, 3}
print(my_set2)  # {1, 2, 3}
my_set1.add(4)
print(my_set1)
my_set1.remove(2)
print(my_set1)
my_set1.update([4,5])
print(my_set1)

# Dictionary - Key-value pairs
my_dict = {"name": "Alice", "age": 25}
print(my_dict["name"])  # Alice
for key in my_dict:
    print(my_dict[key])
