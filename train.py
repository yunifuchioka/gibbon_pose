import argparse
from datetime import datetime

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
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=4)
    parser.add_argument(
        "-e", "--epochs", type=int, help="number of training epochs", default=10
    )
    parser.add_argument(
        "-w", "--n_workers", type=int, help="number of processes", default=8
    )

    args = parser.parse_args()
    if args.name is None:
        args.name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    data_generator = GibGenerator()
    train_generator = TrainingGenerator(data_generator)

    if args.model == "deep_lab_cut":
        model = DeepLabCut(train_generator, subpixel=False)
    # elif args.model == "leap":
    #     model = LEAP(train_generator, subpixel=False)
    elif args.model == "stacked_dense_net":
        model = StackedDenseNet(train_generator, subpixel=False)
    elif args.model == "stacked_hourglass":
        model = StackedHourglass(train_generator, subpixel=False)
    else:
        raise argparse.ArgumentTypeError("invalid model type")

    model.fit(batch_size=args.batch_size, epochs=args.epochs, n_workers=args.n_workers)
    model.save("stats/{}.h5".format(args.name))
