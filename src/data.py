import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def preprocess(x):
    x = np.array(x, dtype=np.float32) / 255.0
    if x.ndim == 2:
        x = x[np.newaxis, ...]
    if x.ndim == 3:
        x = x[..., np.newaxis]
    return x


def load_fashion_mnist(seed=42, val_size=6000):
    (x_all_train, y_all_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    # split 60k train -> 54k train + 6k val, stratified so each class is balanced
    x_train, x_val, y_train, y_val = train_test_split(
        x_all_train,
        y_all_train,
        test_size=val_size,
        stratify=y_all_train,
        random_state=seed,
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    print("Loading Fashion-MNIST ...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fashion_mnist()

    print("\n=== Shapes and dtypes ===")
    print(f"x_train : {x_train.shape}  dtype={x_train.dtype}")
    print(f"y_train : {y_train.shape}  dtype={y_train.dtype}")
    print(f"x_val   : {x_val.shape}  dtype={x_val.dtype}")
    print(f"y_val   : {y_val.shape}  dtype={y_val.dtype}")
    print(f"x_test  : {x_test.shape}  dtype={x_test.dtype}")
    print(f"y_test  : {y_test.shape}  dtype={y_test.dtype}")

    print("\n=== Pixel value range (raw uint8) ===")
    print(f"x_train  min={x_train.min()}, max={x_train.max()}")
    print(f"x_val    min={x_val.min()},   max={x_val.max()}")
    print(f"x_test   min={x_test.min()},  max={x_test.max()}")

    print("\n=== Per-class counts ===")
    header = f"{'Class':<20} {'Train':>7} {'Val':>6} {'Test':>6}"
    print(header)
    print("-" * len(header))
    for label, name in enumerate(CLASS_NAMES):
        tr = int((y_train == label).sum())
        va = int((y_val   == label).sum())
        te = int((y_test  == label).sum())
        print(f"{name:<20} {tr:>7} {va:>6} {te:>6}")

    print("\n=== Preprocess smoke-test ===")
    x_batch = preprocess(x_train[:4])
    print(f"batch of 4  -> shape={x_batch.shape}, dtype={x_batch.dtype}, min={x_batch.min():.3f}, max={x_batch.max():.3f}")
    x_single = preprocess(x_train[0])
    print(f"single img  -> shape={x_single.shape}, dtype={x_single.dtype}, min={x_single.min():.3f}, max={x_single.max():.3f}")

    print("\n[OK] Phase 1 data check complete.")
