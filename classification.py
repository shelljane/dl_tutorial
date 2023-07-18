# Import libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load MNIST dataset
data_path = "data/"
if not os.path.exists(data_path): 
    os.mkdir(data_path)
batch_size_train = 100 # Size of a batch of data for training
batch_size_test = 100 # Size of a batch of data for testing
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

learning_rate = 0.01
# Instantiate the model
network = Net()
# Instantiate the optimizer
optimizer = optim.SGD(network.parameters(), lr=learning_rate)

test_losses = []
accuracies = []
n_epochs = 10
for epoch in range(n_epochs): 
    # Training
    network.train()
    progress = tqdm(train_loader)
    for step, (data, target) in enumerate(progress):
        # Inference
        output = network(data)
        # Compute the loss
        loss = F.cross_entropy(output, target)
        # Gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Log the result
        progress.set_postfix(loss=loss.item())
    # Testing
    test_loss = 0
    accuracy = 0
    correct = 0
    count = 0
    network.eval()
    progress = tqdm(test_loader)
    for step, (data, target) in enumerate(progress):
        with torch.no_grad():
            # Inference
            output = network(data)
            # Compute the loss
            test_loss += F.cross_entropy(output, target).item()
            # Get the prediction
            pred = output.max(dim=1)[1]
            # Count correct predictions
            correct += (pred == target).sum()
            count += target.shape[0]
        # Log the result
        progress.set_postfix(loss=loss.item(), accu=(correct/count).item())
        test_loss /= len(test_loader.dataset)
        accuracy = correct/count
        test_losses.append(test_loss)
        accuracies.append(accuracy)
    print(f"[Epoch {epoch}]: loss={test_loss}, accuracy={accuracy}")

