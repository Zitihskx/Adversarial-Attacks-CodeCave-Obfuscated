import os
import csv

# Specify the directory path
directory = '/home/user/Desktop/Malconv2/TrainingData/TestMalware/'

# List all files in the directory
file_list = os.listdir(directory)

# Specify the CSV file path
csv_file = 'file_list.csv'

# Write the file list to the CSV file
with open(csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['File Name'])  # Write header row
    
    for file_name in file_list:
        csvwriter.writerow([file_name])