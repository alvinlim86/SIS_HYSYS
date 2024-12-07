import pandas as pd
import numpy as np
import tensorflow as tf
import keras as keras
from keras import layers, models

# Generating a linear dataset
np.random.seed(0)  # For reproducibility
data = {
    'feature1': np.linspace(0, 10, 100),
    'feature2': np.linspace(0, 20, 100),
    'target': np.linspace(0, 10, 100) * 2 + np.random.normal(0, 0.1, 100)  # Linear relationship with 10% noise
}
df = pd.DataFrame(data)

print(df.head())

# Convert DataFrame to TensorFlow Dataset
features = df[['feature1', 'feature2']].values
target = df['target'].values

dataset = tf.data.Dataset.from_tensor_slices((features, target))

model = models.Sequential([
    layers.Input(shape=(2,)),  # Two input features
    layers.Dense(10, activation='relu'),  # Hidden layer with 10 neurons
    layers.Dense(1)  # Output layer with one neuron
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Using Mean Squared Error as loss


# Splitting the dataset into training and testing
train_size = int(0.8 * len(features))
train_dataset = dataset.take(train_size).batch(10)
test_dataset = dataset.skip(train_size).batch(10)

# Training the model
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)

# Making predictions on the test dataset
test_features, test_targets = next(iter(test_dataset.unbatch().batch(len(features) - train_size)))
predictions = model.predict(test_features)

# Printing Predictions and Actual Values
print("Predictions vs Actual Values:")
for pred, actual in zip(predictions, test_targets):
    print(f"Predicted: {pred[0]:.2f}, Actual: {actual:.2f}")

# Calculating Mean Absolute Error (MAE) 
mae = keras.metrics.mean_absolute_error(test_targets, predictions)

print("Mean Absolute Error (MAE):") 
for i, mae in enumerate(mae, 1):
    print(f"[{i}]: {mae:.8f}")