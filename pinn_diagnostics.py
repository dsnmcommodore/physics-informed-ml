
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
EPOCHS = 5000
LEARNING_RATE = 0.001
LAMBDA_PHYSICS = 100.0

MASS = 1.0
DAMPING = 0.5
STIFFNESS = 4.0

os.makedirs("results", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------
# Load simulated oscillator data
# --------------------------------------------------

data = pd.read_csv("data/oscillator_data.csv")

time = data["time"].to_numpy(dtype=np.float32)
displacement = data["displacement"].to_numpy(dtype=np.float32)

split_index = int(0.8 * len(time))

# Training observations: 0–approximately 8 seconds
t_train = torch.tensor(
    time[:split_index].reshape(-1, 1),
    dtype=torch.float32
)
x_train = torch.tensor(
    displacement[:split_index].reshape(-1, 1),
    dtype=torch.float32
)

# Evaluation interval: approximately 8–10 seconds
t_test = torch.tensor(
    time[split_index:].reshape(-1, 1),
    dtype=torch.float32
)
x_test = displacement[split_index:]

# Physics collocation points cover the full domain.
t_physics = torch.linspace(0, 10, 300).reshape(-1, 1)

# Initial conditions
t0 = torch.tensor([[0.0]], dtype=torch.float32)
x0 = torch.tensor([[1.0]], dtype=torch.float32)
v0 = torch.tensor([[0.0]], dtype=torch.float32)


# --------------------------------------------------
# Neural network
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


# --------------------------------------------------
# Physics residual
# m*x'' + c*x' + k*x = 0
# --------------------------------------------------

def calculate_physics_residual(model, t):
    t = t.clone().detach().requires_grad_(True)

    x = model(t)

    dx_dt = torch.autograd.grad(
        outputs=x,
        inputs=t,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    d2x_dt2 = torch.autograd.grad(
        outputs=dx_dt,
        inputs=t,
        grad_outputs=torch.ones_like(dx_dt),
        create_graph=True
    )[0]

    residual = (
        MASS * d2x_dt2
        + DAMPING * dx_dt
        + STIFFNESS * x
    )

    return residual


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model(use_physics):
    torch.manual_seed(SEED)

    model = NeuralNetwork()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # Observed-data loss
        prediction = model(t_train)
        data_loss = torch.mean((prediction - x_train) ** 2)

        # Initial position and velocity constraints
        initial_position_loss = torch.mean(
            (model(t0) - x0) ** 2
        )

        t0_grad = t0.clone().detach().requires_grad_(True)
        x0_prediction = model(t0_grad)

        initial_velocity = torch.autograd.grad(
            outputs=x0_prediction,
            inputs=t0_grad,
            grad_outputs=torch.ones_like(x0_prediction),
            create_graph=True
        )[0]

        initial_velocity_loss = torch.mean(
            (initial_velocity - v0) ** 2
        )

        # Physics loss is enabled only for the PINN.
        if use_physics:
            residual = calculate_physics_residual(
                model, t_physics
            )
            physics_loss = torch.mean(residual ** 2)
            loss = (
                data_loss
                + LAMBDA_PHYSICS * physics_loss
                + initial_position_loss
                + initial_velocity_loss
            )
        else:
            physics_loss = torch.tensor(0.0)
            loss = (
                data_loss
                + initial_position_loss
                + initial_velocity_loss
            )

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(
                f"{'PINN' if use_physics else 'Data-only NN'} | "
                f"Epoch {epoch:4d} | "
                f"Total loss: {loss.item():.6f} | "
                f"Data loss: {data_loss.item():.6f} | "
                f"Physics loss: {physics_loss.item():.6f}"
            )

    return model


print("\nTraining data-only neural network...")
data_model = train_model(use_physics=False)

print("\nTraining physics-informed neural network...")
pinn_model = train_model(use_physics=True)


# --------------------------------------------------
# Predictions and test metrics
# --------------------------------------------------

with torch.no_grad():
    data_prediction = (
        data_model(torch.tensor(time.reshape(-1, 1))).numpy().flatten()
    )
    pinn_prediction = (
        pinn_model(torch.tensor(time.reshape(-1, 1))).numpy().flatten()
    )

test_start = split_index

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mae, rmse

data_mae, data_rmse = calculate_metrics(
    x_test, data_prediction[test_start:]
)
pinn_mae, pinn_rmse = calculate_metrics(
    x_test, pinn_prediction[test_start:]
)

print("\nHELD-OUT INTERVAL METRICS (approximately 8–10 s)")
print(f"Data-only NN: MAE={data_mae:.6f}, RMSE={data_rmse:.6f}")
print(f"PINN:         MAE={pinn_mae:.6f}, RMSE={pinn_rmse:.6f}")


# --------------------------------------------------
# Figure 1: Full trajectory and held-out interval
# --------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    time, displacement,
    label="Reference solution", linewidth=2
)
plt.plot(
    time, data_prediction,
    label="Data-only NN", linestyle="--"
)
plt.plot(
    time, pinn_prediction,
    label="PINN", linestyle="-."
)

plt.axvline(
    time[split_index],
    color="gray",
    linestyle=":",
    label="Start of held-out interval"
)

plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Oscillator Predictions: Data-only NN vs PINN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/diagnostic_predictions.png", dpi=300)
plt.show()


# --------------------------------------------------
# Figure 2: Zoom into held-out interval
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    time[split_index:], x_test,
    label="Reference solution", linewidth=2
)
plt.plot(
    time[split_index:], data_prediction[split_index:],
    label="Data-only NN", linestyle="--"
)
plt.plot(
    time[split_index:], pinn_prediction[split_index:],
    label="PINN", linestyle="-."
)

plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Held-out Interval: Approximately 8–10 Seconds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/diagnostic_test_interval.png", dpi=300)
plt.show()


# --------------------------------------------------
# Figure 3: Physics residual across the full domain
# --------------------------------------------------

pinn_residual = calculate_physics_residual(
    pinn_model,
    torch.tensor(time.reshape(-1, 1), dtype=torch.float32)
).detach().numpy().flatten()

plt.figure(figsize=(10, 5))

plt.plot(time, pinn_residual, label="PINN physics residual")
plt.axhline(0, color="gray", linestyle=":")
plt.xlabel("Time (s)")
plt.ylabel("Residual (m/s²)")
plt.title("PINN Governing-Equation Residual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/diagnostic_physics_residual.png", dpi=300)
plt.show()


# --------------------------------------------------
# Save predictions and metrics
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "time": time,
    "reference_displacement": displacement,
    "data_only_prediction": data_prediction,
    "pinn_prediction": pinn_prediction,
    "pinn_physics_residual": pinn_residual
})

prediction_results.to_csv(
    "results/diagnostic_predictions.csv",
    index=False
)

metrics = pd.DataFrame([
    {
        "Model": "Data-only NN",
        "MAE": data_mae,
        "RMSE": data_rmse
    },
    {
        "Model": "PINN",
        "MAE": pinn_mae,
        "RMSE": pinn_rmse
    }
])

metrics.to_csv("results/diagnostic_metrics.csv", index=False)

print("\nSaved diagnostic outputs:")
print("results/diagnostic_predictions.png")
print("results/diagnostic_test_interval.png")
print("results/diagnostic_physics_residual.png")
print("results/diagnostic_predictions.csv")
print("results/diagnostic_metrics.csv")
