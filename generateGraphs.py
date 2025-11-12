import matplotlib.pyplot as plt

files = ["LandClass/rezultati/AlexNet.txt",
         "LandClass/rezultati/ResNet.txt",
         "LandClass/rezultati/VGG.txt"
]

all_epochs = []
all_losses = []

for file in files:
    with open(file, "r") as f:
        lines = f.readlines()
        epochs = []
        losses = []

        for i in range(0, len(lines), 2):
            epoch_line = lines[i]
            parts = epoch_line.split(",")
            epoch = int(parts[0].split(" ")[1])
            loss = float(parts[1].split(": ")[1])
            epochs.append(epoch)
            losses.append(loss)

    all_epochs.append(epochs)
    all_losses.append(losses)

plt.figure(figsize=(10, 6))
i = 0
for file in files:
    plt.plot(all_epochs[i], all_losses[i], marker="o", label=file.split("/")[2].split(".")[0])
    i += 1

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()