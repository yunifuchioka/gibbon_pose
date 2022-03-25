from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import StackedDenseNet, DeepLabCut, LEAP, StackedHourglass

from gib_data_generator import GibGenerator

data_generator = GibGenerator()
train_generator = TrainingGenerator(data_generator)
model = StackedHourglass(train_generator, subpixel=False)
model.fit(batch_size=2, n_workers=8)
model.save("initial_model.h5")
