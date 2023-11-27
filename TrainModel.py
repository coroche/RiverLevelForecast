import torch
from torch import nn
from torch.utils.data import Dataset
import GetTrainingData
from torch.utils.data import DataLoader

class RiverDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = len(self.data[0]) -1
        inputs = torch.tensor(self.data[idx][:x], dtype=torch.float32)
        label = torch.tensor([self.data[idx][x]], dtype=torch.float32)
        return inputs, label
    
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, sampleLength):
        size = sampleLength*3 - 1
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def createDataSet(sampleLength: int) -> RiverDataset:
    data = []
    df = GetTrainingData.getRainLevelDF()
    for i in range(0, len(df.index) - sampleLength):
        sample_df = GetTrainingData.getSample(df, i, sampleLength)
        flat_sample = sample_df.to_numpy().flatten().tolist()
        data.append(flat_sample)
    data = [x for x in data if x]
    dataset = RiverDataset(data)
    return dataset


def splitTrainingAndTestData(dataset: RiverDataset, ratio: float):
    total_samples = len(dataset)
    train_size = int(ratio * total_samples)
    test_size = total_samples - train_size

    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    return training_data, test_data

def createModel(sampleLength: int, lr: float):
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork(sampleLength).to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    return model, device, loss_fn, optimizer

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 5 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":

    sampleLength = 31
    batch_size = 2
    epochs = 1000
    lr = 0.0001
    
    data = createDataSet(sampleLength)
    training_data, test_data = splitTrainingAndTestData(data, 0.8)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    model, device, loss_fn, optimizer = createModel(sampleLength, lr)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    ans = input("Do you want to save this model? (yes/no): ")

    if ans == "yes":

        modeldata = {
            'model_state_dict': model.state_dict(),
            'sampleLength': sampleLength,
            'device': device
        }
        
        torch.save(modeldata, "Data/Dargle/model.pth")
        print("Saved PyTorch Model State to model.pth")
    
    else:
        print("Model disguarded.")


