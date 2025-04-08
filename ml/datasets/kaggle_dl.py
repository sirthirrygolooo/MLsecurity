import kagglehub
import shutil
import os

path = kagglehub.dataset_download("sarahtaha/1025-pokemon")

print("Path to dataset files:", path)
for file in os.listdir(path):
    if file.endswith(".csv"):
        shutil.copy(os.path.join(path, file), os.getcwd())
        shutil.rmtree(os.path.join(path, file), ignore_errors=True)
        print(f"Copied {file} to {os.getcwd()}")
    else:
        print(f"Skipped {file} as it is not a CSV file")

