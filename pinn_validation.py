
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ==================================================
# 1. SETTINGS
# ==================================================

os.makedirs("results", exist_ok=True)

DATA_FRACTIONS = [1.0, 0.5, 0.25, 0.10]
SEEDS = [1, 2, 3]

EPOCHS = 5000
LAMBDA_PHYSICS = 100

MASS = 1.0
DAMPING = 0.5
STIFFNESS = 4.0


# ==================================================
# 2. LOAD DATA
# ==================================================

data = pd.read_csv("data/oscillator_data.csv")

time = data["time"].to_numpy(dtype=np.float32)
x_true = data["displacement"].to_numpy(dtype=np.float32)

split_index = int(len(data) * 0.8)

time_train = time[:split_index]
x_train = x_true[:split_index]

time_test = time[split_index:]
x_test = x_true[split_index:]

t_test = torch.tensor(
    time_test, dtype=torch.float32
).reshape(-1, 1)


# ==================================================
# 3. NEURAL NETWORK
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
# 4. PHYSICS RESIDUAL
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

    return (
        MASS * d2x_dt2
        + DAMPING * dx_dt
        + STIFFNESS * x
    )


# ==================================================
# 5. TRAIN MODEL
# ==================================================

def train_model(
    t_observed,
    x_observed,
    lambda_physics,
    seed
):

    torch.manual_seed(seed)

    model = NeuralNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    t_physics = torch.linspace(
        0, 10, 300
    ).reshape(-1, 1)

    t0 = torch.tensor([[0.0]], dtype=torch.float32)
    x0 = torch.tensor([[1.0]], dtype=torch.float32)

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        # Fit observed displacement data
        prediction = model(t_observed)

        data_loss = torch.mean(
            (prediction - x_observed) ** 2
        )

        # Enforce the governing differential equation
        residual = physics_residual(
            model, t_physics
        )

        physics_loss = torch.mean(
            residual ** 2
        )

        # Initial displacement: x(0) = 1
        initial_position_loss = torch.mean(
            (model(t0) - x0) ** 2
        )

        # Initial velocity: x'(0) = 0
        t0_grad = t0.clone().detach().requires_grad_(True)
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
# 6. METRICS
# ==================================================

def calculate_metrics(actual, predicted):

    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


# ==================================================
# 7. REPEATED VALIDATION
# ==================================================

results = []

for fraction in DATA_FRACTIONS:

    number_of_points = max(
        2, int(len(time_train) * fraction)
    )

    for seed in SEEDS:

        print(
            f"\nData fraction: {fraction:.0%} | "
            f"Seed: {seed}"
        )

        # Reproducibly sample observations.
        # Always include the initial observation.
        rng = np.random.default_rng(seed)

        selected = rng.choice(
            np.arange(1, len(time_train)),
            size=number_of_points - 1,
            replace=False
        )

        selected = np.sort(
            np.concatenate(([0], selected))
        )

        t_observed = torch.tensor(
            time_train[selected],
            dtype=torch.float32
        ).reshape(-1, 1)

        x_observed = torch.tensor(
            x_train[selected],
            dtype=torch.float32
        ).reshape(-1, 1)

        # Same sampled data and seed for both models
        for model_name, lambda_value in [
            ("Data-only NN", 0),
            ("PINN", LAMBDA_PHYSICS)
        ]:

            model = train_model(
                t_observed,
                x_observed,
                lambda_value,
                seed
            )

            # Autograd is needed during PINN training,
            # but not for ordinary prediction.
            with torch.no_grad():
                prediction = (
                    model(t_test).numpy().flatten()
                )

            mae, mse, rmse = calculate_metrics(
                x_test, prediction
            )

            results.append({
                "Data_fraction": fraction,
                "Observations": number_of_points,
                "Seed": seed,
                "Model": model_name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse
            })

            print(
                f"{model_name}: "
                f"MAE={mae:.5f}, RMSE={rmse:.5f}"
            )


# ==================================================
# 8. SAVE INDIVIDUAL TRIALS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/validation_trials.csv",
    index=False
)


# ==================================================
# 9. CALCULATE MEAN AND STANDARD DEVIATION
# ==================================================

summary = (
    results_df
    .groupby(["Data_fraction", "Model"])
    .agg(
        Mean_MAE=("MAE", "mean"),
        SD_MAE=("MAE", "std"),
        Mean_RMSE=("RMSE", "mean"),
        SD_RMSE=("RMSE", "std")
    )
    .reset_index()
)

summary.to_csv(
    "results/validation_summary.csv",
    index=False
)

print("\n\nVALIDATION SUMMARY")
print(summary.to_string(index=False))


# ==================================================
# 10. PLOT MEAN RMSE WITH STANDARD DEVIATION
# ==================================================

plt.figure(figsize=(9, 6))

for model_name in ["Data-only NN", "PINN"]:

    subset = summary[
        summary["Model"] == model_name
    ].sort_values("Data_fraction")

    plt.errorbar(
        subset["Data_fraction"] * 100,
        subset["Mean_RMSE"],
        yerr=subset["SD_RMSE"],
        marker="o",
        capsize=4,
        label=model_name
    )

plt.xlabel("Training observations used (%)")
plt.ylabel("RMSE on unseen 8–10 s interval")
plt.title("Repeated-Run Validation of PINN and Neural Network")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/validation_rmse.png",
    dpi=300
)

plt.show()

print("\nSaved:")
print("results/validation_trials.csv")
print("results/validation_summary.csv")
print("results/validation_rmse.png")
