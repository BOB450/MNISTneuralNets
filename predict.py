import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import argparse
import sys


# Define the CNNModel class matching the trained model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully Connected Layer
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        # Output Layer with Softmax Activation
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def load_model(model_path, device):
    """
    Loads the trained CNN model from the specified file.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: Loaded CNN model in evaluation mode.
    """
    model = CNNModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from '{model_path}'.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)
    return model


def preprocess_image(image_path):
    """
    Preprocesses the input image to match the MNIST dataset format.

    Steps:
        1. Convert to grayscale.
        2. Invert colors (MNIST digits are white on black).
        3. Resize to 28x28 pixels.
        4. Convert to tensor and normalize.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for prediction.
    """
    try:
        # Open the image file
        image = Image.open(image_path).convert('L')  # Convert to grayscale
    except Exception as e:
        print(f"Error opening image file: {e}")
        sys.exit(1)

    # Invert the image (MNIST digits are white on black)
    image = ImageOps.invert(image)

    # Resize to 28x28 pixels using the appropriate resampling filter
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS

    image = image.resize((28, 28), resample=resample_filter)

    # Define the same transforms as during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST
    ])

    # Apply the transforms
    image = transform(image)

    # Add a batch dimension
    image = image.unsqueeze(0)  # Shape: [1, 1, 28, 28]

    return image


def predict(model, image, device):
    """
    Predicts the digit in the preprocessed image using the loaded model.

    Args:
        model (nn.Module): Loaded CNN model.
        image (torch.Tensor): Preprocessed image tensor.
        device (torch.device): Device to perform computation on.

    Returns:
        tuple: Probabilities and corresponding digit classes.
    """
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        probs, classes = probabilities.topk(10, dim=1)
        return probs.cpu().numpy()[0], classes.cpu().numpy()[0]


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MNIST Digit Prediction using a Trained CNN Model')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    parser.add_argument('--model', type=str, default='mnist_cnn.pth',
                        help='Path to the saved CNN model file (default: mnist_cnn.pth)')
    args = parser.parse_args()

    # Determine the device to run on (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load the trained model
    model = load_model(args.model, device)

    # Preprocess the input image
    image = preprocess_image(args.image_path)
    print('Image preprocessed successfully.')

    # Perform prediction
    probabilities, classes = predict(model, image, device)
    print('Prediction complete.\n')

    # Display the results
    print('Digit Probabilities:')
    for i in range(10):
        print(f'Digit {classes[i]}: {probabilities[i] * 100:.2f}%')


if __name__ == '__main__':
    main()
