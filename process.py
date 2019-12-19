import os
import cv2
import h5py
import numpy as np

path = "data/images/"


images = []
files = os.listdir(path)
for i, file in enumerate(files):
    try:
        img = cv2.imread(path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        if i % 100 == 0:
            print("{}/{}".format(i, len(files)))
    except:
        print(file + " is corrupted!")
images = np.stack(images)
with h5py.File("data/data.hdf5") as file:
    file.create_dataset("images", data=images)
