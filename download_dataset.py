from roboflow import Roboflow
import os

# Initialize Roboflow with your API key
rf = Roboflow(api_key="eyCISGeaViIzawW8JHrp")

# Specify the project and version (replace with your version code, e.g., "ppe-gloves/1")
project = rf.workspace("object-detection-vs3ra").project("glove-q7czq-slsbk")
version = project.version(1)  # Replace 1 with your version number

# Download dataset in YOLOv8 format
dataset = version.download("yolov8")
print(f"Dataset downloaded to: {dataset.location}")



# import zipfile
# import os

# zip_path = "glove-1/roboflow.zip"  # Replace with your zip file path
# extract_path = "glove-1"     # Replace with desired folder
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)
# print(f"Extracted to {extract_path}")