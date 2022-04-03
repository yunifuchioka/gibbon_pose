import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")
plt.rcParams["font.size"] = "40"

textfile = open("results/04-01-2022_dlc_gc4/output.txt", "r")
lines = textfile.readlines()

epoch = -4

epochs = []
losses = []
val_losses = []

for idx, line in enumerate(lines):
    if "val_loss" in line:
        no_newline = line.replace("\n", "")
        line_split = no_newline.split("loss: ")
        loss = line_split[1].split(" - ")[0]
        val_loss = line_split[2]

        epochs.append(epoch)
        losses.append(float(loss))
        val_losses.append(float(val_loss))
        epoch += 1

epochs = np.array(epochs)
losses = np.array(losses)
val_losses = np.array(val_losses)

save_array = np.vstack(
    (epochs[epochs >= 0], losses[epochs >= 0], val_losses[epochs >= 0])
)

plt.plot(save_array[0, :], save_array[1, :])
plt.plot(save_array[0, :], save_array[2, :])
plt.title("DeepLabCut Model Training Progress")
plt.legend(["loss", "validation loss"])
plt.xlabel("epochs")
plt.ylabel("MSE error")
plt.show()
