import torch
import onnx
import onnxruntime as ort
import numpy as np
from os import path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

if not path.exists("data/cv_model.pth"):
    
    
    batch_size = 64
    
    # Create data loader.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break
    
    
    model = NeuralNetwork().to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def test(dataloader, model):
        size = len(dataloader.dataset)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
    print("Done!")
    
    
    torch.save(model.state_dict(), "data/cv_model.pth")
    print("Saved PyTroch Model State to cv_model.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("data/cv_model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# Export the model using `torch.onnx.export`
torch.onnx.export(model,                               # model being run
                  torch.randn(1, 28, 28),              # model input (or a tuple for multiple inputs)
                  "data/fashion_mnist_model.onnx",     # where to save the model (can be a file or file like object)
                  input_names = ['input'],             # the model's input names
                  output_names = ['output'])           # the model's output names

# Load the onnx model with `onnx.load`
onnx_model = onnx.load("data/fashion_mnist_model.onnx")
onnx.checker.check_model(onnx_model)

# Create inference session using `ort.InferenceSession`
ort_sess = ort.InferenceSession('data/fashion_mnist_model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
outputs = ort_sess.run(None, {'input': x.numpy()})

# Print Result
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')