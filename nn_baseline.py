import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
#I add this to allow reproducibility 
torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

data = pd.read_csv("data/oscillator_data.csv")

time = data["time"].values
displacement = data["displacement"].values


# --------------------------------------------------
# 2. Chronological train/test split
# --------------------------------------------------

split_index = int(len(data) * 0.8)

time_train = time[:split_index]
x_train = displacement[:split_index]

time_test = time[split_index:]
x_test = displacement[split_index:]


# Convert training data to tensors

t_train = torch.tensor(
    time_train,
    dtype=torch.float32
).reshape(-1, 1)

x_train = torch.tensor(
    x_train,
    dtype=torch.float32
).reshape(-1, 1)


# --------------------------------------------------
# 3. Neural network
# --------------------------------------------------

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.network(t)


model = NeuralNetwork()


# --------------------------------------------------
# 4. Loss and optimizer
# --------------------------------------------------

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 5. Training
# --------------------------------------------------

epochs = 5000

for epoch in range(epochs):

    optimizer.zero_grad()

    prediction = model(t_train)

    loss = loss_function(
        prediction,
        x_train
    )

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch:5d} | "
            f"Loss: {loss.item():.6f}"
        )


# --------------------------------------------------
# 6. Predict entire time range
# --------------------------------------------------

t_all = torch.tensor(
    time,
    dtype=torch.float32
).reshape(-1, 1)

with torch.no_grad():

    prediction = model(t_all)

prediction = prediction.numpy().flatten()


# --------------------------------------------------
# 7. Evaluate unseen 8–10 s region
# --------------------------------------------------

test_prediction = prediction[split_index:]

mae = np.mean(
    np.abs(x_test - test_prediction)
)

mse = np.mean(
    (x_test - test_prediction) ** 2
)

rmse = np.sqrt(mse)


print("\nData-only Neural Network performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)


# --------------------------------------------------
# 8. Plot
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.scatter(
    time,
    displacement,
    label="Physics data"
)

plt.plot(
    time,
    prediction,
    label="Neural Network"
)

plt.axvline(
    x=8,
    linestyle="--",
    label="Train/Test boundary"
)

plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Data-Only Neural Network")
plt.grid(True)
plt.legend()

plt.show()