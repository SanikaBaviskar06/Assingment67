import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Loan data
X = np.array([
    [250000, 600, 200000, 15000, 0],
    [400000, 700, 300000, 9000, 1],
    [600000, 750, 500000, 12000, 1],
    [200000, 550, 150000, 15000, 0],
    [600000, 800, 700000, 16000, 1],
    [350000, 650, 250000, 9000, 1],
    [180000, 500, 150000, 12000, 0],
    [500000, 850, 900000, 15000, 1],
    [300000, 600, 200000, 14000, 0],
    [700000, 780, 600000, 16000, 1]
])

# Output values
y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create neural network
model = Sequential()

model.add(Dense(8, activation="relu", input_shape=(5,)))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Train model
model.fit(X_train, y_train, epochs=50, verbose=1)


# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy :", accuracy)


# Test new applicant
new_applicant = np.array([[135000, 720, 400000, 10300, 1]])

new_applicant = scaler.transform(new_applicant)

prediction = model.predict(new_applicant)

if prediction[0][0] >= 0.5:
    print("Prediction: Loan Approved")
else:
    print("Prediction: Loan Rejected")