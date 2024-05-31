import os
import sys

def read_last_line(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
    return None

def process_directory(directory):
    output_file = os.path.join(directory, 'numbers.txt')
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                last_line = read_last_line(file_path)
                if last_line:
                    outfile.write(f'{last_line}\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_numbers.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    process_directory(directory)
