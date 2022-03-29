import numpy as np
from tensorflow import keras

from deepposekit.models import load_model
from deepposekit.io import VideoReader

from gib_data_generator import GibGenerator
from utils import *

if __name__ == "__main__":
    model = load_model("results/dlc_40.h5")

    generator = GibGenerator()

    indices = np.random.randint(low=0, high=len(generator), size=10)
    for idx in indices:
        img, gt_pose = generator[idx]
        predictions = model.predict_on_batch(img)
        pose = predictions[0, :, :2]
        plot_img(img[0, :, :, :], keypoint_array2dict(pose))
        plt.title(idx)
        plt.show()
