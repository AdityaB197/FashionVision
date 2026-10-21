"""CNN model for Fashion-MNIST classification."""

import tensorflow as tf


def build_cnn():
    """Build and return a compiled CNN for Fashion-MNIST.

    Architecture:
        Conv -> Pool -> Conv -> Pool -> Dropout -> Dense -> Dropout -> Dense
    """
    model = tf.keras.Sequential([
        # Input layer: 28x28 greyscale images with 1 channel
        tf.keras.layers.Input(shape=(28, 28, 1)),

        # Conv2D learns local patterns (edges, textures) by sliding small filters
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),

        # MaxPooling reduces spatial size, keeps the strongest activations
        tf.keras.layers.MaxPooling2D(),

        # Second conv block: learns higher-level patterns from the first
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        # Dropout randomly zeros 25% of neurons during training to reduce overfitting
        tf.keras.layers.Dropout(0.25),

        # Flatten converts the 2D feature maps into a 1D vector for the dense layers
        tf.keras.layers.Flatten(),

        # Dense layer combines all features for classification
        tf.keras.layers.Dense(128, activation="relu"),

        # Higher dropout (50%) before the output layer — most effective here
        tf.keras.layers.Dropout(0.5),

        # Output: 10 neurons, one per class; softmax gives probabilities that sum to 1
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
