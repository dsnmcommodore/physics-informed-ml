import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ==================================================
# 1. REPRODUCIBILITY
# ==================================================

torch.manual_seed(42)
np.random.seed(42)

os.makedirs("results", exist_ok=True)


# ==================================================
# 2. LOAD THE PHYSICAL DATA
# ==================================================

data = pd.read_csv("data/oscillator_data.csv")

time = data["time"].to_numpy()
x_true = data["displacement"].to_numpy()

split_index = int(len(data) * 0.8)

time_train = time[:split_index]
x_train = x_true[:split_index]

time_test = time[split_index:]
x_test = x_true[split_index:]


# ==================================================
# 3. PHYSICAL PARAMETERS
# ==================================================

m = 1.0
c = 0.5
k = 4.0


# ==================================================
# 4. NEURAL NETWORK
# ==================================================

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


# ==================================================
# 5. PHYSICS RESIDUAL
# ==================================================

def physics_residual(model, t):

    t = t.clone().detach().requires_grad_(True)

    x = model(t)

    dx_dt = torch.autograd.grad(
        x,
        t,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    d2x_dt2 = torch.autograd.grad(
        dx_dt,
        t,
        grad_outputs=torch.ones_like(dx_dt),
        create_graph=True
    )[0]

    return m * d2x_dt2 + c * dx_dt + k * x


# ==================================================
# 6. TRAINING FUNCTION
# ==================================================

def train_model(
    t_train,
    x_train,
    lambda_physics,
    seed=42,
    epochs=5000
):

    torch.manual_seed(seed)

    model = NeuralNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Physics constraints are evaluated across
    # the complete time domain.
    t_physics = torch.linspace(
        0,
        10,
        300
    ).reshape(-1, 1)

    # Known initial conditions
    t0 = torch.tensor(
        [[0.0]],
        dtype=torch.float32
    )

    x0_true = torch.tensor(
        [[1.0]],
        dtype=torch.float32
    )

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
        initial_position_loss = torch.mean(
            (model(t0) - x0_true) ** 2
        )

        # Initial velocity
        t0_grad = (
            t0.clone()
            .detach()
            .requires_grad_(True)
        )

        x0_prediction = model(t0_grad)

        v0_prediction = torch.autograd.grad(
            x0_prediction,
            t0_grad,
            grad_outputs=torch.ones_like(x0_prediction),
            create_graph=True
        )[0]

        initial_velocity_loss = torch.mean(
            v0_prediction ** 2
        )

        # Total loss
        loss = (
            data_loss
            + lambda_physics * physics_loss
            + initial_position_loss
            + initial_velocity_loss
        )

        loss.backward()
        optimizer.step()

    return model


# ==================================================
# 7. EVALUATION
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
# 8. SPARSE-DATA EXPERIMENT
# ==================================================

data_fractions = [1.0, 0.5, 0.25, 0.10]

lambda_physics = 100

results = []

predictions_for_plot = {}

# Use a separate random seed for sampling the
# observations, so the sampling is reproducible.
sampling_rng = np.random.default_rng(42)

for fraction in data_fractions:

    print("\n" + "=" * 55)
    print(f"Training data fraction: {fraction:.0%}")
    print("=" * 55)

    # Select observations from the training interval.
    # Keep the initial observation at t = 0.
    number_of_points = max(
        2,
        int(len(time_train) * fraction)
    )

    selected_indices = np.sort(
        sampling_rng.choice(
            np.arange(1, len(time_train)),
            size=number_of_points - 1,
            replace=False
        )
    )

    selected_indices = np.concatenate(
        ([0], selected_indices)
    )

    sampled_time = time_train[selected_indices]
    sampled_x = x_train[selected_indices]

    t_train_tensor = torch.tensor(
        sampled_time,
        dtype=torch.float32
    ).reshape(-1, 1)

    x_train_tensor = torch.tensor(
        sampled_x,
        dtype=torch.float32
    ).reshape(-1, 1)

    print(f"Training observations: {len(sampled_time)}")

    # ------------------------------
    # A. DATA-ONLY NEURAL NETWORK
    # ------------------------------

    data_model = train_model(
        t_train_tensor,
        x_train_tensor,
        lambda_physics=0,
        seed=42
    )

    # ------------------------------
    # B. PHYSICS-INFORMED NN
    # ------------------------------

    pinn = train_model(
        t_train_tensor,
        x_train_tensor,
        lambda_physics=lambda_physics,
        seed=42
    )

    # ------------------------------
    # Predictions on unseen interval
    # ------------------------------

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
            pinn(t_test_tensor)
            .numpy()
            .flatten()
        )

    data_metrics = calculate_metrics(
        x_test,
        data_prediction
    )

    pinn_metrics = calculate_metrics(
        x_test,
        pinn_prediction
    )

    results.append({
        "Data_fraction": fraction,
        "Observations": len(sampled_time),
        "Model": "Data-only NN",
        "MAE": data_metrics[0],
        "MSE": data_metrics[1],
        "RMSE": data_metrics[2]
    })

    results.append({
        "Data_fraction": fraction,
        "Observations": len(sampled_time),
        "Model": "PINN",
        "MAE": pinn_metrics[0],
        "MSE": pinn_metrics[1],
        "RMSE": pinn_metrics[2]
    })

    predictions_for_plot[fraction] = {
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
    "results/sparse_data_experiment.csv",
    index=False
)

print("\nFINAL RESULTS")
print(results_df.to_string(index=False))


# ==================================================
# 10. PLOT RMSE AGAINST DATA FRACTION
# ==================================================

plt.figure(figsize=(9, 6))

for model_name in ["Data-only NN", "PINN"]:

    subset = results_df[
        results_df["Model"] == model_name
    ].sort_values("Data_fraction")

    plt.plot(
        subset["Data_fraction"] * 100,
        subset["RMSE"],
        marker="o",
        label=model_name
    )

plt.xlabel("Training observations used (%)")
plt.ylabel("RMSE on unseen 8–10 s interval")

plt.title("Effect of Training Data Availability")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/sparse_data_experiment.png",
    dpi=300
)

plt.show()


# ==================================================
# 11. PLOT PREDICTIONS WITH ONLY 10% OF DATA
# ==================================================

fraction = 0.10

predictions = predictions_for_plot[fraction]

plt.figure(figsize=(11, 6))

plt.plot(
    time,
    x_true,
    label="True physical solution",
    linewidth=2
)

plt.scatter(
    time_train,
    x_train,
    label="Available training observations",
    s=12
)

plt.plot(
    time_test,
    predictions["data_only"],
    label="Data-only NN prediction"
)

plt.plot(
    time_test,
    predictions["pinn"],
    label="PINN prediction"
)

plt.axvline(
    x=8,
    linestyle="--",
    label="Train/test boundary"
)

plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")

plt.title("Prediction with 10% of Training Observations")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/sparse_data_predictions.png",
    dpi=300
)

plt.show()