import os

file_path = 'gestures/pray.csv'

if os.path.exists(file_path):
    os.remove(file_path)
    print("Pickle file deleted.")
else:
    print("File does not exist.")
