import pandas as pd
import matplotlib.pyplot as plt

#loads the dataset
data = pd.read_csv("data/oscillator_data.csv")
print(data.head())

print("Dataset information:")
print(data.info())

print("Statistical summary:")
print(data.describe())

print("Missing values:")
print(data.isnull().sum())

#to visualize the data
plt.figure(figsize=(10,5))
plt.plot(data["time"], data["displacement"], label="Displacement")
plt.plot(data["time"], data["velocity"], label="Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("Physical System Variables")
plt.grid(True)
plt.legend()
plt.show()

'''x = data[["time"]] #Scikit-learn accepts data in 2D
y = data["displacement"]

#this is the cool part, the ml training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=10), LinearRegression())

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print("Model performance:")
print(f"MAE:  {mae:.6f}")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")

plt.figure(figsize=(10, 5))

plt.scatter(x_test,y_test,label="Physics data",s=15)

plt.scatter( x_test,  y_pred, label="ML prediction",s=15)
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Physics vs Machine Learning")
plt.grid(True)
plt.legend()
plt.show()'''

# Define input and target
x = data[["time"]]
y = data["displacement"]

# Chronological train/test split
# First 80% of the time → training
# Last 20% of the time → testing

split_index = int(len(data) * 0.8)

x_train = x.iloc[:split_index]
x_test = x.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training time range:")
print(x_train["time"].min(), "to", x_train["time"].max())

print("\nTesting time range:")
print(x_test["time"].min(), "to", x_test["time"].max())

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model = make_pipeline(PolynomialFeatures(degree=10),LinearRegression())

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

plt.figure(figsize=(10, 5))

plt.scatter(x_test["time"],y_test,label="Physics data")

plt.scatter(x_test["time"],y_pred,label="ML prediction")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Chronological ML Prediction vs Physics")
plt.grid(True)
plt.legend()
plt.show()