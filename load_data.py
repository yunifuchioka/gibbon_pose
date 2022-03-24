import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

from utils import *

# desired image sizes found through manually looking through all gibbon images and
# determining max width and height
des_img_height = 500
des_img_width = 500

df_raw = pd.read_json("dataset/train_annotation.json", orient="split")
df_gib = df_raw[df_raw.species == "Gibbon"]  # extract only gibbon entries
df_gib = df_gib.reset_index()  # sets indices specific to gibbon only dataframe
num_data = df_gib.shape[0]  # number of images

# choose a gibbon image randomly
index = random.randint(0, num_data)
print(index)

curr = df_gib.loc[index]
curr_fname = curr["file"]
curr_bbox = np.array(curr["bbox"])
curr_vis = np.array(curr["visibility"])
curr_img = np.array(Image.open("dataset/train/" + curr_fname))

# determine image size and pad accordingly
curr_img_height, curr_img_width, _ = curr_img.shape
top_pad = np.floor((des_img_height - curr_img_height) / 2).astype(np.uint16)
bottom_pad = np.ceil((des_img_height - curr_img_height) / 2).astype(np.uint16)
right_pad = np.ceil((des_img_width - curr_img_width) / 2).astype(np.uint16)
left_pad = np.floor((des_img_width - curr_img_width) / 2).astype(np.uint16)
curr_img = np.pad(
    curr_img,
    ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
    mode="constant",
    constant_values=0,
)

# get joint landmark locations and convert to dictionary
curr_pose_array = np.array(curr["landmarks"]).reshape((17, 2))
curr_pose = {}
for landmark in Landmarks:
    curr_pose[landmark.name] = curr_pose_array[landmark.value, :]

# adjust landmark locations according to pad amount
curr_pose_array[:, 1] += top_pad
curr_pose_array[:, 0] += left_pad


plot_img(curr_img, curr_pose)
plt.show()
