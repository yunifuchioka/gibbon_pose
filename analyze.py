import numpy as np
from tensorflow import keras

from deepposekit.models import load_model
from deepposekit.io import VideoReader

from gib_data_generator import GibGenerator
from utils import *


def analyze_train_set(model):
    generator = GibGenerator()

    indices = np.random.randint(low=0, high=len(generator), size=10)
    for idx in indices:
        img, gt_pose = generator[idx]
        predictions = model.predict_on_batch(img)
        pose = predictions[0, :, :2]
        plot_img_pred(img[0, :, :, :], predictions[0, :, :])
        plt.title(idx)
        plt.show()


def analyze_video(model):
    generator = VideoReader("videos/brach-2D-crop.mp4", batch_size=1)
    fig = plt.figure()

    def draw_frame(k):
        fig.clear()
        img = generator[k]
        predictions = model.predict_on_batch(img)
        pose = predictions[0, :, :2]
        plot_img_pred(img[0, :, :, :], predictions[0, :, :])

    import matplotlib.animation as animation

    anim = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(generator) // 2,
        interval=0.033 * 1000.0,
        repeat=True,
        blit=False,
    )

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=33.3, metadata=dict(artist="Me"), bitrate=1000)
    anim.save("videos/brach-2D-pred.mp4", writer=writer)

    plt.show()


if __name__ == "__main__":
    model = load_model("results/dlc_40.h5")
    # model = load_model("results/03-24-2022_23-44-25.h5")
    # model = load_model("results/dense_init.h5")

    # analyze_train_set(model)
    analyze_video(model)
