# 🧬 Quantum Classification with PennyLane

This project demonstrates a simple *quantum machine learning classifier* using the [PennyLane](https://pennylane.ai/) library.  
We train a *variational quantum circuit* to perform *binary classification* on the Iris dataset.

---

## 🧠 What Is This About?

Quantum machine learning (QML) combines quantum computing with data science.  
Here, we build a quantum neural network that:
- Embeds classical features as quantum states
- Applies variational layers (trainable parameters)
- Predicts a class label from quantum measurement

> 🔗 *Learn more about QML*  
> - [Introduction to PennyLane](https://docs.pennylane.ai/en/stable/introduction/introduction.html)  
> - [PennyLane Iris Classifier Tutorial](https://pennylane.ai/qml/demos/tutorial_quantum_classifier.html)

---

## 🗃 Dataset Used

We use the *Iris dataset* from scikit-learn.  
To simplify the task, we reduce it to a *binary classification* problem by filtering only classes 0 and 1.

---

## 🧪 Quantum Model Details

- *4 qubits* represent the 4 features  
- *AngleEmbedding* encodes data into the quantum state  
- *BasicEntanglerLayers* form the trainable circuit  
- *Measurement* on the first qubit yields an expectation value  

> 📌 We optimize the circuit parameters using *gradient descent* to minimize the *mean squared error* between predicted expectation values and target labels.

---

## ▶ How to Run

### 📦 Prerequisites

- Python 3.7+  
- PennyLane  
- scikit-learn  

### 🛠 Install dependencies

```bash
pip install pennylane scikit-learn
```

🚀 Run the script
```bash
python quantum_classifier.py
```

You will see the training cost decrease and, at the end, an accuracy score printed:
```
Cost: 0.4921
…
Accuracy: 0.95
```

---

📂 Project Structure
```
.
├── quantum_classifier.py   # Quantum classification script
└── README.md               # This file
```

---

📚 Learn More

📘 PennyLane Documentation

🎓 Qiskit Textbook: Machine Learning

🧠 PennyLane Variational Classifier Tutorial


---

🙋‍♂ Author

Created as a learning project using PennyLane.
Feel free to fork, contribute, or experiment with other datasets!
