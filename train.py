import argparse
import os
from datetime import datetime
import gc
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback


from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import DeepLabCut, LEAP, StackedDenseNet, StackedHourglass

from gib_data_generator import GibGenerator

if __name__ == "__main__":

    # get experiment name
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="experiment name", default=None)
    parser.add_argument(
        "-m", "--model", type=str, help="network model", default="deep_lab_cut"
    )
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=2)
    parser.add_argument(
        "-e", "--epochs", type=int, help="number of training epochs", default=100
    )
    parser.add_argument(
        "-w", "--n_workers", type=int, help="number of processes", default=8
    )
    parser.add_argument(
        "-v", "--valid_split", type=float, help="validation split", default=0.2
    )

    args = parser.parse_args()
    if args.name is None:
        args.name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    save_dir = "results/{}/".format(args.name)
    # create results dir if not there already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_generator = GibGenerator()
    train_generator = TrainingGenerator(
        data_generator, validation_split=args.valid_split
    )

    if args.model == "deep_lab_cut":
        model = DeepLabCut(train_generator)
    elif args.model == "leap":
        # need to overwrite train_generator with 0 downsample factor for LEAP
        train_generator = TrainingGenerator(
            data_generator, downsample_factor=0, validation_split=args.valid_split
        )
        model = LEAP(train_generator)
    elif args.model == "stacked_dense_net":
        model = StackedDenseNet(train_generator)
    elif args.model == "stacked_hourglass":
        model = StackedHourglass(train_generator)
    else:
        raise argparse.ArgumentTypeError("invalid model type")

    # https://github.com/tensorflow/tensorflow/issues/31312#issuecomment-821809246
    class ClearMemory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            k.clear_session()

    callbacks_list = [ClearMemory()]

    model.compile(optimizer="adam", loss="mse", run_eagerly=True)

    for epoch in range(args.epochs):
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        model.fit(
            batch_size=args.batch_size,
            validation_batch_size=args.batch_size,
            callbacks=callbacks_list,
            epochs=1,
            n_workers=args.n_workers,
        )
        if (epoch + 1) % 10 == 0:
            model.save(save_dir + "epoch-{}.h5".format(epoch))
