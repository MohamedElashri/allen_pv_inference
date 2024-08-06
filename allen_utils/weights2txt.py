# Script to convert pvfinder weights to format expected by PVFinder Kernel.


import csv
import os

def csv_to_txt(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.csv', '.txt'))

            with open(input_path, 'r') as csvfile, open(output_path, 'w') as txtfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    txtfile.write(' '.join(row) + '\n')

            print(f"Converted {filename} to txt format.")

# Usage
input_directory = '/data/home/melashri/iris/debug-allen/non-Allen/closure_test/model_parameters'
output_directory = '/data/home/melashri/iris/debug-allen/non-Allen/closure_test/model_parameters/model_parameters_txt'
csv_to_txt(input_directory, output_directory)
