import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

from utils import *


df_raw = pd.read_json("dataset/train_annotation.json", orient="split")
df_gib = df_raw[df_raw.species == "Gibbon"]  # extract only gibbon entries
df_gib = df_gib.reset_index()  # sets indices specific to gibbon only dataframe
num_data = df_gib.shape[0]  # number of images

index = random.randint(0, num_data)
print(index)

curr = df_gib.loc[index]
curr_fname = curr["file"]
curr_bbox = np.array(curr["bbox"])
curr_vis = np.array(curr["visibility"])
curr_img = Image.open("dataset/train/" + curr_fname)

# get joint landmark locations and convert to dictionary
curr_pose_array = np.array(curr["landmarks"]).reshape((17, 2))
curr_pose = {}
for landmark in Landmarks:
    curr_pose[landmark.name] = curr_pose_array[landmark.value, :]

plot_img(curr_img, curr_pose)
plt.show()