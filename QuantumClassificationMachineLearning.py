import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Define a quantum device with 4 wires to match the number of features
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def quantum_model(features, weights):
    # Embed the input features into the quantum circuit
    qml.templates.AngleEmbedding(features, wires=range(4))
    # Apply layers of entangling gates and rotation gates
    qml.templates.BasicEntanglerLayers(weights, wires=range(4))
    # Measure the expectation value of the Z operator on the first qubit
    return qml.expval(qml.PauliZ(wires=0))

# Initialize random weights for the quantum circuit with 4 qubits
num_layers = 6
weights = np.random.random(size=(num_layers, 4))

# Load the iris dataset and preprocess it
iris = load_iris()
X, y = iris.data, iris.target

# Filter out one class to focus on a binary classification problem
is_class_0_or_1 = (y == 0) | (y == 1)
X, y = X[is_class_0_or_1], y[is_class_0_or_1]

# Standardize the features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert class labels to -1 and 1 for binary classification
y_train_mapped = np.where(y_train == 0, -1, 1)

# Define the cost function
def cost_function(weights):
    # Calculate the expectation value for each training example
    exp_vals = np.array([quantum_model(x, weights) for x in X_train])
    # Compute the cost as the mean squared error between
    # the expectation value and the mapped labels
    cost = np.mean((exp_vals - y_train_mapped) ** 2)
    return cost

# Optimize the circuit parameters using a gradient descent optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.5)
for _ in range(30):
    weights, cost = opt.step_and_cost(cost_function, weights)
    print(f"Cost: {cost:.4f}")

# Define a function to make predictions with the trained model
def predict(features, weights):
    # Obtain predictions from the quantum model for each feature set
    predictions = [quantum_model(f, weights) for f in features]
    # Convert predictions to binary labels based on the sign of the output
    return np.where(np.array(predictions) >= 0, 1, 0)

# Predict the labels of the test data
y_pred = predict(X_test, weights)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
