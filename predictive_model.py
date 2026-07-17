import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("student_data.csv")

# Input (Hours) and Output (Marks)
X = df[["Hours"]]
y = df["Marks"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks
predictions = model.predict(X)

# Print predictions
print("Predicted Marks:")
for hour, mark in zip(df["Hours"], predictions):
    print(f"Hours: {hour} -> Predicted Marks: {mark:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(6,4))
plt.scatter(df["Hours"], df["Marks"], label="Actual Data")
plt.plot(df["Hours"], predictions, label="Prediction")
plt.title("Hours vs Marks Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.tight_layout()

plt.savefig("prediction_chart.png")
plt.show()