import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")
plt.rcParams["font.size"] = "40"

# textfile = open("results/04-01-2022_dlc_gc4/output.txt", "r")
textfile = open("results/04-03-2022_sdn/output.txt", "r")

# output_string_type = "deeplabcut"
output_string_type = "stacked_dense_net"

# set starting value to nonzero value if there is some extra stuff in the beginning of the string
# epoch = -4
epoch = 0

lines = textfile.readlines()

epochs = []
losses = []
val_losses = []

for idx, line in enumerate(lines):
    if "val_loss" in line:
        no_newline = line.replace("\n", "")
        if output_string_type == "deeplabcut":
            line_split = no_newline.split("loss: ")
            loss = line_split[1].split(" - ")[0]
            val_loss = line_split[2]
        elif output_string_type == "stacked_dense_net":
            line_split = no_newline.split(" - ")
            loss = line_split[2].split("loss: ")[1]
            val_loss = line_split[5].split("loss: ")[1]

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

# manually remove point after index 66, where something weird happens
if output_string_type == "stacked_dense_net":
    save_array = save_array[:, :60]

plt.plot(save_array[0, :], save_array[1, :])
plt.plot(save_array[0, :], save_array[2, :])
# plt.title("DeepLabCut Model Training Progress")
plt.title("Stacked Dense Net Training Progress")
plt.legend(["loss", "validation loss"])
plt.xlabel("epochs")
plt.ylabel("MSE error")
plt.show()
