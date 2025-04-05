# The with keyword in Python is used to create context managers that
# handle resource management and exception safety. It ensures proper
# acquisition and release of resources (like files, locks, or network connections) 
# even if errors occur.

# Without 'with'
file = open('data.txt', 'w')
try:
    file.write('Hello')
finally:          # finally will catch any error and runs its block
    file.close()  # Manual cleanup

# With 'with' (recommended)
with open('data.txt', 'w') as file:
    file.write('Hello')
    # automatic closing and clean-up

