import os

def list_directory_structure(start_path, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(start_path):
            level = root.replace(start_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            f.write(f'{indent}{os.path.basename(root)}/\n')
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f'{sub_indent}{file}\n')

start_path = '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024'  # Change this to your project directory path
output_file = 'directory_structure.txt'
list_directory_structure(start_path, output_file)

print(f'Directory structure has been saved to {output_file}')
