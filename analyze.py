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
    generator = VideoReader("videos/brach-2D-crop.mp4", batch_size=1)
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
        frames=len(generator) // 2,
        interval=0.033 * 1000.0,
        repeat=True,
        blit=False,
    )

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=33.3, metadata=dict(artist="Me"), bitrate=1000)
    anim.save("videos/brach-2D-04-03-2022_sdn_epoch-59_thresh-07.mp4", writer=writer)

    plt.show()


def compare_training_set(model1, model2):
    generator = GibGenerator()

    num_images = 5
    indices = np.random.randint(low=0, high=len(generator), size=num_images)

    fig, ax = plt.subplots(2, num_images, sharex=True, sharey=True, figsize=(15, 7))
    fig.subplots_adjust(wspace=0)
    for count, idx in enumerate(indices):
        img, gt_pose = generator[idx]
        predictions = model1.predict_on_batch(img)
        pose = predictions[0, :, :2]
        plt.subplot(2, num_images, count + 1)
        plot_img_pred(img[0, :, :, :], predictions[0, :, :], thresh=0.07)
        plt.axis("off")
        # plt.title(idx)

    for count, idx in enumerate(indices):
        img, gt_pose = generator[idx]
        predictions = model2.predict_on_batch(img)
        pose = predictions[0, :, :2]
        plt.subplot(2, num_images, num_images + count + 1)
        plot_img_pred(img[0, :, :, :], predictions[0, :, :], thresh=0.07)
        plt.axis("off")
        # plt.title(idx)
    plt.show()


def overlay_interp(interp_pose):
    generator = VideoReader("videos/brach-2D-crop.mp4", batch_size=1)
    fig = plt.figure()

    def draw_frame(k):
        fig.clear()
        img = generator[k][:, :, :, [2, 1, 0]]
        pose = interp_pose[k, :, :]
        plot_img_pred(img[0, :, :, :], pose, thresh=0.07)
        print(k)

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
    anim.save("videos/brach-2D-dlc_40_pred_interpolated.mp4", writer=writer)

    plt.show()


if __name__ == "__main__":
    model = load_model("results/04-01-2022_dlc_gc4/epoch-60.h5")
    model2 = load_model("results/04-03-2022_sdn/epoch-59.h5")

    # analyze_train_set(model)
    # analyze_video(model)
    compare_training_set(model, model2)

    # interp_pose = np.load("results/nil/dlc_40_results_pred_interpolated.npy")
    # overlay_interp(interp_pose)
