import torch
import matplotlib.pyplot as plt
import os

os.makedirs("results/accuracy_plots", exist_ok=True)
os.makedirs("results/time_plots", exist_ok=True)

def fgsm_attack(image, epsilon, label, model, criterion):
    if epsilon == 0:
        return image.clone()
    image.requires_grad = True
    output = model(image)
    loss = criterion(output, torch.tensor([label]))
    model.zero_grad()
    loss.backward()
    perturbed = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed, 0, 1)

def plot_metrics(df, message_size):
    epsilons = df["epsilon"]
    accuracies = df["accuracy"]
    times = df["execution_time_sec"]

    os.makedirs("results/accuracy_plots", exist_ok=True)
    os.makedirs("results/time_plots", exist_ok=True)

    plt.figure()
    plt.plot(epsilons, accuracies, marker='o')
    plt.title(f"Accuracy vs Epsilon (Message Size: {message_size})")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(f"results/accuracy_plots/accuracy_msg_{message_size}.png")
    plt.close()

    plt.figure()
    plt.plot(epsilons, times, marker='o', color='red')
    plt.title(f"Execution Time vs Epsilon (Message Size: {message_size})")
    plt.xlabel("Epsilon")
    plt.ylabel("Time (sec)")
    plt.grid(True)
    plt.savefig(f"results/time_plots/time_msg_{message_size}.png")
    plt.close()
