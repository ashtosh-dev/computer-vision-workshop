
"""CNN model architecture."""

import tensorflow as tf


def build_cnn_model(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    number_of_classes: int = 10,
) -> tf.keras.Model:
    """Build a regularized CNN for CIFAR-10 classification."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),

            # Data augmentation
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomRotation(0.05),

            # First convolutional block
            tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),

            # Second convolutional block
            tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.3),

            # Third convolutional block
            tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.GlobalAveragePooling2D(),

            # Classification head
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(number_of_classes, activation="softmax"),
        ],
        name="cifar10_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
