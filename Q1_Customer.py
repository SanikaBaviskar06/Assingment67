import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Customer data
X = np.array([
    [25, 500, 12, 1, 2],
    [30, 700, 24, 0, 1],
    [45, 1300, 6, 5, 8],
    [50, 1500, 5, 6, 10],
    [35, 600, 18, 1, 1],
    [35, 800, 30, 0, 6],
    [45, 1400, 4, 7, 9],
    [52, 1600, 3, 8, 12],
    [27, 500, 30, 0, 1],
    [42, 1500, 8, 4, 7]
])

# Output values
y = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])


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


# Test new customer
new_customer = np.array([[44, 1450, 5, 6, 9]])

new_customer = scaler.transform(new_customer)

prediction = model.predict(new_customer)

if prediction[0][0] >= 0.5:
    print("Prediction: Customer will leave")
else:
    print("Prediction: Customer will stay")