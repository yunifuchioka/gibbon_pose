import numpy as np
import pandas as pd
from PIL import Image
from deepposekit.io import BaseGenerator, TrainingGenerator

from utils import *


class GibGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        """
        Initializes the class.
        If graph and swap_index are not defined,
        they are set to a vector of -1 corresponding
        to keypoints shape
        """
        df_raw = pd.read_json("dataset/train_annotation.json", orient="split")
        self.df_gib = df_raw[df_raw.species == "Gibbon"]  # extract only gibbon entries
        # sets indices specific to gibbon only dataframe
        self.df_gib = self.df_gib.reset_index()
        self.num_data = self.df_gib.shape[0]  # number of images

        self.graph = (
            np.full(self.compute_keypoints_shape(), -1).flatten().astype(np.int16)
        )
        self.swap_index = (
            np.full(self.compute_keypoints_shape(), -1).flatten().astype(np.int16)
        )

        # load entire gibbon dataset onto memory
        images_array = []
        keypoints_array = []
        for index in range(self.num_data):
            curr = self.df_gib.loc[index]
            curr_img = np.array(Image.open("dataset/train/" + curr["file"]))

            # determine image size and pad accordingly
            img_shape = self.compute_image_shape()
            curr_img_height, curr_img_width, _ = curr_img.shape
            top_pad = np.floor((img_shape[0] - curr_img_height) / 2).astype(np.uint16)
            bottom_pad = np.ceil((img_shape[0] - curr_img_height) / 2).astype(np.uint16)
            right_pad = np.ceil((img_shape[1] - curr_img_width) / 2).astype(np.uint16)
            left_pad = np.floor((img_shape[1] - curr_img_width) / 2).astype(np.uint16)
            curr_img = np.pad(
                curr_img,
                ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            curr_pose_array = np.array(curr["landmarks"]).reshape(
                self.compute_keypoints_shape()
            )
            curr_pose_array[:, 0] += left_pad
            curr_pose_array[:, 1] += top_pad

            images_array.append(curr_img)
            keypoints_array.append(curr_pose_array)
        self.images = np.stack(images_array)
        self.keypoints = np.stack(keypoints_array).astype(np.float64)


    def __len__(self):
        """
        Returns the number of samples in the generator as an integer (int64)
        """
        return self.num_data

    def compute_image_shape(self):
        """
        Returns a tuple of integers describing
        the image shape in the form:
        (height, width, n_channels)
        """
        # set to be the same over all images via zero padding
        return (512, 512, 3)

    def compute_keypoints_shape(self):
        """
        Returns a tuple of integers describing the
        keypoints shape in the form:
        (n_keypoints, 2), where 2 is the x,y coordinates
        """
        return (17, 2)

    def get_images(self, indexes):
        """
        Takes a list or array of indexes corresponding
        to image-keypoint pairs in the dataset.
        Returns a numpy array of images with the shape:
        (n_samples, height, width, n_channels)
        """
        return self.images[indexes,:,:,:]

    def get_keypoints(self, indexes):
        """
        Takes a list or array of indexes corresponding to
        image-keypoint pairs in the dataset.
        Returns a numpy array of keypoints with the shape:
        (n_samples, n_keypoints, 2), where 2 is the x,y coordinates
        """
        return self.keypoints[indexes,:,:]

    def set_keypoints(self, indexes, keypoints):
        """
        Takes a list or array of indexes and corresponding
        to keypoints.
        Sets the values of the keypoints corresponding to the indexes
        in the dataset.
        """
        # we have a fixed dataset, we don't want to support modifying data
        raise NotImplementedError()


if __name__ == "__main__":
    generator = GibGenerator()
    indices = np.random.randint(low=0, high=len(generator), size=10)
    images = generator.get_images(indices)
    keypoints = generator.get_keypoints(indices)

    for img_idx in range(images.shape[0]):
        plot_img(
            images[img_idx, :, :, :], keypoint_array2dict(keypoints[img_idx, :, :])
        )
        plt.show()
