import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp# Physical parameters

m = 1.0       # mass (kg)
c = 0.5       # damping coefficient (Ns/m)
k = 4.0       # spring constant (N/m)

# Initial conditions
x0 = 1.0      # initial displacement (m)
v0 = 0.0      # initial velocity (m/s)

def oscillator(t, y):
    x,v = y
    dxdt = v
    dvdt = -(c/m)*v - (k/m)*x
    return [dxdt, dvdt]

#time range
t_start = 0.0
t_end = 10.0
t = np.linspace(t_start, t_end, 1000)

#Initial state
initial_conditions = [x0, v0]

#solve the differential equations
solution = solve_ivp(oscillator, [t_start, t_end], initial_conditions, t_eval=t)

x = solution.y[0]
v = solution.y[1]

data = pd.DataFrame({"time":t,"displacement":x,"velocity":v})

data.to_csv("data/oscillator_data.csv",index = False)

plt.figure(figsize=(10,5))
plt.plot(t,x,label = "Displacement")
plt.xlabel("Time(s)")
plt.ylabel("Displacement (m)")
plt.title("Damped Harmonic Oscillator")
plt.grid(True)
plt.legend()
plt.show()