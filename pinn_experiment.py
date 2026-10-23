import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn



# for reproducibility
torch.manual_seed(42)
np.random.seed(42)


#to load data
data = pd.read_csv("data/oscillator_data.csv")
time = data["time"].values
displacement = data["displacement"].values


#train/test split

split_index = int(len(data) * 0.8)

time_train = time[:split_index]
x_train = displacement[:split_index]

time_test = time[split_index:]
x_test = displacement[split_index:]


# convert training data to tensors
t_data = torch.tensor(time_train,dtype=torch.float32).reshape(-1, 1)

x_data = torch.tensor( x_train, dtype=torch.float32).reshape(-1, 1)


#physics collocation points
t_physics = torch.linspace( 0, 10, 300).reshape(-1, 1)


#initial conditions
t_initial = torch.tensor( [[0.0]], dtype=torch.float32)

x_initial = torch.tensor( [[1.0]],  dtype=torch.float32)


#neural network
class PINN(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, 32),nn.Tanh(),nn.Linear(32, 32),nn.Tanh(), nn.Linear(32, 32),nn.Tanh(), nn.Linear(32, 1) )

    def forward(self, t):
        return self.network(t)



# physics residual
def physics_residual(model, t):

    t = ( t.clone().detach().requires_grad_(True) )
    x = model(t)

    dx_dt = torch.autograd.grad(x,
        t,grad_outputs=torch.ones_like(x), create_graph=True )[0]

    d2x_dt2 = torch.autograd.grad(dx_dt,t,grad_outputs=torch.ones_like(dx_dt),create_graph=True)[0]

    m = 1.0
    c = 0.5
    k = 4.0

    residual = ( m * d2x_dt2 + c * dx_dt + k * x)

    return residual


#trainig one model
def train_model(lambda_physics):

    # reseting random seed so every model starts from the same initial weights
    torch.manual_seed(42)
    model = PINN()
 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5000

    for epoch in range(epochs):

        optimizer.zero_grad()
        
        prediction = model(t_data)

        data_loss = torch.mean(
            (prediction - x_data) ** 2)

        residual = physics_residual(model, t_physics)

        physics_loss = torch.mean(residual ** 2)


        
        # initial position
        x0_prediction = model(t_initial)
        initial_position_loss = torch.mean( (x0_prediction - x_initial) ** 2)


        
        # initial velocity

        t0 = (t_initial.clone().detach().requires_grad_(True) )

        x0 = model(t0)

        v0_prediction = torch.autograd.grad(  x0, t0, grad_outputs=torch.ones_like(x0), create_graph=True)[0]

        initial_velocity_loss = torch.mean( v0_prediction ** 2)

        #total loss

        loss = (data_loss + lambda_physics * physics_loss + initial_position_loss + initial_velocity_loss)

        loss.backward()
        optimizer.step()


    # ----------------------------------------
    # evaluate

    t_all = torch.tensor(time, dtype=torch.float32  ).reshape(-1, 1)

    with torch.no_grad():
        prediction = model(t_all).numpy().flatten()


    # Unseen 8–10 second region

    test_prediction = prediction[split_index:]

    mae = np.mean( np.abs(x_test - test_prediction))

    mse = np.mean( (x_test - test_prediction) ** 2)

    rmse = np.sqrt(mse)


    return mae, mse, rmse, prediction


# --------------------------------------------------
# 6. Run experiment
# --------------------------------------------------

lambda_values = [ 0,0.01,0.1, 1,  10,  100]

results = []
predictions = {}


for lambda_value in lambda_values:
    print(f"\nTraining model with "f"lambda = {lambda_value}")
    mae, mse, rmse, prediction = train_model(lambda_value)
    results.append({"lambda": lambda_value, "MAE": mae,"MSE": mse,"RMSE": rmse})
    predictions[lambda_value] = prediction
    print( f"MAE:  {mae:.6f}")
    print( f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")



#results table
results_df = pd.DataFrame(results)

print("\n")
print(results_df.to_string(index=False))


#I have to save resuts
results_df.to_csv(
    "results/pinn_lambda_experiment.csv",index=False)

#ploting RMSE vs physics weight
plt.figure(figsize=(8, 5))
plt.plot( results_df["lambda"],results_df["RMSE"],marker="o")
plt.xscale("symlog", linthresh=0.01)
plt.xlabel("Physics weight (lambda)")
plt.ylabel("RMSE on 8–10 s")
plt.title("Effect of Physics Constraint Strength")
plt.grid(True)
plt.show()