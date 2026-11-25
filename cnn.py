
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Reshape Images
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 4. Create CNN Model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 5. Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train Model
model.fit(x_train, y_train, epochs=5)

# 7. Evaluate Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_accuracy)

# 8. Predict Image
prediction = model.predict(x_test[:1])

print("Predicted Digit:", prediction.argmax())
print("Actual Digit:", y_test[0])

# 9. Display Image
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title("Predicted Digit: " + str(prediction.argmax()))
plt.show()
