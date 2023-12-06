import sys
import re

if len(sys.argv) != 2:
    print("Usage: python rename_extension.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

# Check if the file exists
try:
    with open(filename, 'r') as file:
        data = file.read()
except FileNotFoundError:
    print(f"File '{filename}' not found.")
    sys.exit(1)

# Use regular expressions to replace '.gif' with '.txt'

new_data = re.sub(r'\.gif', '.txt', data)

# Write the modified data back to the file
with open(filename, 'w') as file:
    file.write(new_data)

print(f"All occurrences of '.gif' in '{filename}' have been replaced with '.txt'.")

