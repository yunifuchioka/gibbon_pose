from deepposekit.io.TrainingGenerator import TrainingGenerator
import numpy as np
from tensorflow import keras

from deepposekit.models import load_model
from deepposekit.io import VideoReader

from gib_data_generator import GibGenerator
from utils import *


def analyze_train_set(model):
    base_generator = GibGenerator()
    # training generator supports shuffling
    generator = TrainingGenerator(base_generator)

    indices = np.random.randint(low=0, high=len(generator), size=10)
    for idx in indices:
        img, gt_pose = generator[idx]
        predictions = model.predict_on_batch(img)
        pose = predictions[0, :, :2]
        plot_img_pred(img[0, :, :, :], predictions[0, :, :])
        plt.title(idx)
        plt.show()


def analyze_video(model):
    generator = VideoReader("videos/brach-naples-crop.mp4", batch_size=1)
    fig = plt.figure()

    def draw_frame(k):
        fig.clear()
        img = generator[k][:, :, :, [2, 1, 0]]
        predictions = model.predict_on_batch(img)
        pose = predictions[0, :, :2]
        plot_img_pred(img[0, :, :, :], predictions[0, :, :], thresh=0.07)
        print(k)

    import matplotlib.animation as animation

    anim = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(generator) // 3,
        interval=0.033 * 1000.0,
        repeat=True,
        blit=False,
    )

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=33.3, metadata=dict(artist="Me"), bitrate=1000)
    anim.save(
        "videos/brach-naples-04-01-2022_dlc_gc4_epoch-90_thresh-07.mp4", writer=writer
    )

    plt.show()


if __name__ == "__main__":
    model = load_model("results/04-01-2022_dlc_gc4/epoch-90.h5")

    # analyze_train_set(model)
    analyze_video(model)
