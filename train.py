import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Enable cuDNN benchmark mode
torch.backends.cudnn.benchmark = True

# Smish Activation Function
class Smish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))

# CNN Model Architecture with Smish Activation
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.smish = Smish()
        # Layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        # Layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.4)
        # Fully Connected Layer
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        # Output Layer
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Layer 1
        x = self.smish(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        # Layer 2
        x = self.smish(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        # Layer 3
        x = self.smish(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully Connected Layer
        x = self.smish(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        # Output Layer with Log-Softmax for classification
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


if __name__ == '__main__':
    # Data augmentation and normalization
    train_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transforms)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model, define loss function, and optimizer
    model = CNNModel().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters())

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')

    # Evaluation on test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    test_loss /= len(test_loader)
    test_acc = 100 * correct / len(test_dataset)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print('Model saved to mnist_cnn.pth')
