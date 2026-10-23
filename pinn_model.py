import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
#for reproducibility
torch.manual_seed(42)
np.random.seed(42)

#load physical data
data = pd.read_csv("data/oscillator_data.csv")
time = data["time"].values
displacement = data["displacement"].values
split_index = int(len(data)*0.8)
time_train = time[:split_index]
x_train = displacement[:split_index]

#This covert to pytorch tensors
t_data = torch.tensor(time_train, dtype = torch.float32).reshape(-1,1)
x_data = torch.tensor(x_train, dtype = torch.float32).reshape(-1,1)

#establishing the neural network
class PINN(nn.Module):
    def __init__(self):
     super().__init__()
     self.network = nn.Sequential(nn.Linear(1,32),nn.Tanh(), nn.Linear(32,32),nn.Tanh(), nn.Linear(32,32),nn.Tanh(), nn.Linear(32,1))

    def forward(self, t):
     return self.network(t)

#creates model
model = PINN()
#print(model)


def physics_residuel(model,t):
   #creates a tensor that tracks the derivatives 
   t = t.clone().detach().requires_grad_(True)
   #makes sure PyTorch tracks derivatives with respect to time
   t.requires_grad_(True)
   # what the neural network is to predict
   x = model(t)
   #first derivative prediction:
   dx_dt = torch.autograd.grad(x,t,grad_outputs=torch.ones_like(x),create_graph=True)[0]
   # second derivative
   d2x_dt2 = torch.autograd.grad(dx_dt,t,grad_outputs=torch.ones_like(dx_dt),create_graph=True)[0]

    #physical parameters
   m = 1.0
   c = 0.5
   k = 4.0
#the equation(the physics equation of the system)
   residual = (m* d2x_dt2 + c * dx_dt + k * x)
   return residual

#testing with some time values
t_physics = torch.linspace(0,10,100).reshape(-1,1)
#residual = physics_residual(model,t_test)
#print("Residual shape", residual.shape)
#print("Average absolute residual:",torch.mean(torch.abs(residual)).item())
   

#intial conditions
t_initial = torch.tensor([[0.0]], dtype = torch.float32)
x_initial = torch.tensor([[1.0]], dtype = torch.float32)

#acting optimizer(most things are optimize this day haha)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

#training session
epochs = 5000
for epoch in range(epochs):
   optimizer.zero_grad()
   #In case I confuse myself for data loss and data loss
   x_prediction = model(t_data)
   data_loss = torch.mean(x_prediction - x_data) **2

   residual = physics_residuel(model,t_physics)
   physics_loss = torch.mean(residual**2)

   x0_prediction = model(t_initial)
   intial_position_loss = torch.mean(x0_prediction - x_initial) **2

   t_initial_derivtive = (t_initial.clone().detach().requires_grad_(True))
   x0 = model(t_initial_derivtive)
   v0_prediction  = torch.autograd.grad(x0,t_initial_derivtive,grad_outputs = torch.ones_like(x0), create_graph = True)[0]
   initial_velocity_loss = torch.mean(v0_prediction**2)

   #total loss
   loss = (data_loss + physics_loss + intial_position_loss + initial_velocity_loss)
   #backp
   loss.backward()
   optimizer.step()

   if epoch % 500 == 0:
      print(f"Epoch {epoch:5d}|" f"Total:{loss.item():.6f}|" f"Data:{data_loss.item():.6f}|" f"Physics:{physics_loss.item():.6f}")


#aim of this is to make predictions across the entire time range
t_all = torch.tensor(time, dtype = torch.float32).reshape(-1,1)

with torch.no_grad():
   x_pinn = model(t_all).numpy().flatten()

test_time = time[split_index:]
test_actual = displacement[split_index:]
test_prediction = x_pinn[split_index:]

mae = np.mean(np.abs(test_actual - test_prediction))
mse = np.mean((test_actual-test_prediction)**2)
rmse = np.sqrt(mse)

print("PINN performance on unseen 8-10 s data:")
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)


#ploting results
plt.figure(figsize=(10,5))
plt.scatter(time,displacement,label="Physics data")
plt.plot(time,x_pinn,label="PINN prediction")
plt.axvline(x=8,linestyle="--",label="Train/Test boundary")

plt.xlabel("Time(s)")
plt.ylabel("Displacement(m)")
plt.title("Physics-Informed Neural Network")
plt.grid(True)
plt.legend()
plt.show()


#above I had to get it right because it is the heart of the physics informed model



   

