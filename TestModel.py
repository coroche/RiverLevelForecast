import TrainModel
import random

modeldata = TrainModel.torch.load("Data/Dargle/model.pth")
sampleLength = modeldata['sampleLength']
device = modeldata['device']

model = TrainModel.NeuralNetwork(sampleLength).to(device)
model.load_state_dict(modeldata['model_state_dict'])

model.eval()

data = TrainModel.createDataSet(sampleLength)
i = random.randint(0, len(data))
x, y = data[i][0], data[i][1]
with TrainModel.torch.no_grad():
    x = x.to(device)
    pred = model(x).item()
    predicted, actual = pred, y.item()
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
