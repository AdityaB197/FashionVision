"""MobileNetV2 transfer learning model architecture for FashionVision v2.

Transfer learning leverages rich visual representations (edges, textures, patterns)
learned by MobileNetV2 on 1.4 million ImageNet images, adapting them to real clothing
photographs with minimal compute and dataset size.

Why layers are frozen first:
    The newly added classification head starts with random weights. If all layers
    were trainable from the start, massive initial gradients would backpropagate into
    MobileNetV2 and destroy ("catastrophically overwrite") its pretrained feature extractors.
    Freezing the base allows the head to stabilize first.

Why fine-tuning uses a tiny learning rate:
    When unfreezing the top layers in stage 2, an optimizer with a small learning rate
    (e.g., 1e-5 vs 1e-3) gently nudges higher-level filters to recognize clothing-specific
    textures (collars, hemlines, knit patterns) without destabilizing general visual representations.

Why augmentation helps with real photos:
    Unlike uniform Fashion-MNIST scans, real clothing photographs vary wildly in lighting,
    camera angle, zoom, and folds. Random augmentations force the CNN to learn invariant
    semantic clothing traits rather than memorizing backgrounds or studio setups.
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_real_model(num_classes, img_size=(224, 224)):
    """Build and compile a transfer learning model based on MobileNetV2.

    Input expects float32 RGB images with pixel values in the range [0, 255].
    Because Rescaling(scale=1/127.5, offset=-1) is built directly inside the saved model,
    both training and downstream inference accept standard [0, 255] RGB pixel values,
    guaranteeing that preprocessing cannot diverge between environments.
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3), dtype="float32", name="image_input")

    # Data augmentation layers (automatically active only during training=True)
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.1, name="aug_rotation")(x)
    x = layers.RandomZoom(0.15, name="aug_zoom")(x)
    x = layers.RandomContrast(0.1, name="aug_contrast")(x)
    x = layers.RandomBrightness(0.1, name="aug_brightness")(x)

    # Built-in normalization: maps [0, 255] -> [-1.0, 1.0] (matching MobileNetV2 preprocess_input).
    # Using Rescaling rather than Lambda guarantees seamless native Keras serialization.
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="input_rescaling")(x)

    # Base pretrained feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size[0], img_size[1], 3),
        name="mobilenetv2_base",
    )
    base_model.trainable = False

    # Keep base_model in inference mode to freeze BatchNorm statistics
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    # dtype="float32" on the output layer ensures numerical stability with mixed precision.
    # Softmax in float16 can overflow (exp() has a narrow range in float16).
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fashionvision_v2_mobilenet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def enable_fine_tuning(model, num_layers=30):
    """Unfreeze the top num_layers of the base model while keeping BatchNorm frozen.

    Recompiles the model with a conservative learning rate (1e-5) for fine-tuning.
    """
    # Locate the MobileNetV2 base model inside the functional graph
    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers") and ("mobilenet" in layer.name.lower() or isinstance(layer, tf.keras.Model)):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not locate MobileNetV2 base model within the provided model.")

    base_model.trainable = True

    # Freeze all lower layers up to the last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    # For the top num_layers, unfreeze weights but keep BatchNormalization layers frozen.
    # Updating moving mean/variance on small fine-tuning batches can destabilize representations.
    for layer in base_model.layers[-num_layers:]:
        if isinstance(layer, layers.BatchNormalization) or "batchnorm" in layer.name.lower():
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
