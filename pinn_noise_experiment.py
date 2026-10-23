import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn



# for reproducibility


torch.manual_seed(42)
np.random.seed(42)

os.makedirs("results", exist_ok=True)


# load data

data = pd.read_csv("data/oscillator_data.csv")

time = data["time"].to_numpy()
x_true = data["displacement"].to_numpy()

split_index = int(len(data) * 0.8)

time_train = time[:split_index]
x_train_true = x_true[:split_index]

time_test = time[split_index:]
x_test_true = x_true[split_index:]



#physics parameters
m = 1.0
c = 0.5
k = 4.0



# neural network definition
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(1, 32), nn.Tanh(),nn.Linear(32, 32),nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, t):
      t_normalized = (t - 5.0) / 5.0
      return self.network(t_normalized)


# physical residual

def physics_residual(model, t):

    t = t.clone().detach().requires_grad_(True)

    x = model(t)

    dx_dt = torch.autograd.grad(x,t, grad_outputs=torch.ones_like(x),create_graph=True )[0]

    d2x_dt2 = torch.autograd.grad(dx_dt,t,grad_outputs=torch.ones_like(dx_dt),create_graph=True )[0]

    residual = (m * d2x_dt2+ c * dx_dt+ k * x )

    return residual


# ==================================================
# 6. TRAIN ONE MODEL
# ==================================================

def train_model(
    t_train,
    x_train,
    lambda_physics,
    epochs=5000
):

    model = NeuralNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Physics collocation points over the full domain
    t_physics = torch.linspace(
        0,
        10,
        300
    ).reshape(-1, 1)

    # Known initial conditions
    t0 = torch.tensor([[0.0]], dtype=torch.float32)
    x0_true = torch.tensor([[1.0]], dtype=torch.float32)

    for epoch in range(epochs):

        optimizer.zero_grad()

        # Data loss
        prediction = model(t_train)

        data_loss = torch.mean(
            (prediction - x_train) ** 2
        )

        # Physics loss
        residual = physics_residual(
            model,
            t_physics
        )

        physics_loss = torch.mean(
            residual ** 2
        )

        # Initial position
        initial_position = torch.mean(
            (model(t0) - x0_true) ** 2
        )

        # Initial velocity
        t0_derivative = (
            t0.clone()
            .detach()
            .requires_grad_(True)
        )

        x0_prediction = model(t0_derivative)

        v0_prediction = torch.autograd.grad(
            x0_prediction,
            t0_derivative,
            grad_outputs=torch.ones_like(x0_prediction),
            create_graph=True
        )[0]

        initial_velocity = torch.mean(
            v0_prediction ** 2
        )

        # Total loss
        loss = (data_loss + lambda_physics * physics_loss  + 10.0 * initial_position + 10.0 * initial_velocity)
        loss.backward()
        optimizer.step()

    return model


# ==================================================
# 7. EVALUATION METRICS
# ==================================================

def calculate_metrics(actual, predicted):

    mae = np.mean(
        np.abs(actual - predicted)
    )

    mse = np.mean(
        (actual - predicted) ** 2
    )

    rmse = np.sqrt(mse)

    return mae, mse, rmse


# ==================================================
# 8. RUN NOISE EXPERIMENTS
# ==================================================

noise_levels = [0.02, 0.05, 0.10]

lambda_physics = 1

results = []

example_predictions = {}


for noise_level in noise_levels:

    print("\n" + "=" * 50)
    print(f"Noise standard deviation: {noise_level} m")
    print("=" * 50)

    # Generate noisy training observations.
    # Both models receive the same observations.
    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0.0,
        scale=noise_level,
        size=len(x_train_true)
    )

    x_train_noisy = x_train_true + noise

    # Convert training data to tensors
    t_train_tensor = torch.tensor(
        time_train,
        dtype=torch.float32
    ).reshape(-1, 1)

    x_train_tensor = torch.tensor(
        x_train_noisy,
        dtype=torch.float32
    ).reshape(-1, 1)

    # ----------------------------------------------
    # A. DATA-ONLY NEURAL NETWORK
    # ----------------------------------------------

    torch.manual_seed(42)

    data_model = train_model(
        t_train_tensor,
        x_train_tensor,
        lambda_physics=0
    )

    # ----------------------------------------------
    # B. PHYSICS-INFORMED NEURAL NETWORK
    # ----------------------------------------------

    torch.manual_seed(42)

    pinn_model = train_model(
        t_train_tensor,
        x_train_tensor,
        lambda_physics=lambda_physics
    )

    # Predict the unseen test interval
    t_test_tensor = torch.tensor(
        time_test,
        dtype=torch.float32
    ).reshape(-1, 1)

    with torch.no_grad():

        data_prediction = (
            data_model(t_test_tensor)
            .numpy()
            .flatten()
        )

        pinn_prediction = (
            pinn_model(t_test_tensor)
            .numpy()
            .flatten()
        )

    # Evaluate against the original clean solution
    data_metrics = calculate_metrics(
        x_test_true,
        data_prediction
    )

    pinn_metrics = calculate_metrics(
        x_test_true,
        pinn_prediction
    )

    results.append({
        "Noise_SD": noise_level,
        "Model": "Data-only NN",
        "MAE": data_metrics[0],
        "MSE": data_metrics[1],
        "RMSE": data_metrics[2]
    })

    results.append({
        "Noise_SD": noise_level,
        "Model": "PINN",
        "MAE": pinn_metrics[0],
        "MSE": pinn_metrics[1],
        "RMSE": pinn_metrics[2]
    })

    example_predictions[noise_level] = {
        "data_only": data_prediction,
        "pinn": pinn_prediction
    }

    print(
        f"Data-only NN RMSE: {data_metrics[2]:.6f}"
    )

    print(
        f"PINN RMSE:         {pinn_metrics[2]:.6f}"
    )


# ==================================================
# 9. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/noise_experiment.csv",
    index=False
)

print("\nFINAL RESULTS")
print(results_df.to_string(index=False))


# ==================================================
# 10. PLOT PREDICTIONS AT HIGHEST NOISE LEVEL
# ==================================================

noise_level = 0.10

predictions = example_predictions[noise_level]

plt.figure(figsize=(11, 6))

plt.scatter(
    time_train,
    x_train_true + np.random.default_rng(42).normal(
        0,
        noise_level,
        len(time_train)
    ),
    label="Noisy training observations",
    s=12
)

plt.plot(
    time,
    x_true,
    label="True physical solution",
    linewidth=2
)

plt.plot(
    time_test,
    predictions["data_only"],
    label="Data-only NN prediction"
)

plt.plot(time_test, predictions["pinn"],label="PINN prediction")
plt.axvline( x=8, linestyle="--",label="Train/test boundary")                                   
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Prediction with Noisy Training Data")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/noise_experiment.png",dpi=300)
plt.show() 