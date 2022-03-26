from deepposekit.models import load_model
from deepposekit.io import VideoReader

from tensorflow import keras

from utils import *

model = load_model("results/03-25-2022_07-44-01.h5")

reader = VideoReader('results/train_video.mp4', batch_size=1)
predictions = model.predict(reader)

reader2 = VideoReader('results/train_video.mp4', batch_size=1)
for idx in range(len(reader2)):
    img = reader2.read()
    pose = predictions[idx,:,:2]
    plot_img(img, keypoint_array2dict(pose))
    plt.show()