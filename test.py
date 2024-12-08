import os

data_dir = "D:/Python dev/tomato-leaf-disease-detection/datasets"  # atau gunakan path Anda
if os.path.exists(data_dir):
    print("Path is correct!")
else:
    print("Path does not exist!")