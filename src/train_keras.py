import argparse
import glob
import os

import numpy as np
from model_keras import get_model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


def reshape_y(data: np.ndarray):
    """
    Get gold values from data. multi-hot vector to one-hot vector
    Input:
     - data: ndarray [num_examples x 6]
    Output:
    - ndarray [num_examples]
    """
    return to_categorical(data[:, 5])


def reshape_x(data, dtype=np.float16):
    """
    Get images from data as a list and preprocess them.
    Input:
     - data: ndarray [num_examples x 6]
     -dtype: numpy dtype for the output array
      from the sequence of images
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]
    """
    mean = np.array([0.485, 0.456, 0.406], dtype)
    std = np.array([0.229, 0.224, 0.225], dtype)
    ims_seqs = []
    for i in range(0, len(data)):
        frames = []
        for j in range(0, 5):
            img = np.array(data[i][j], dtype=dtype) / 255
            frames.append((img - mean) / std)
        ims_seqs.append(frames)
    return np.asarray(ims_seqs)


def load_file(path: str, fp: int = 16) -> (np.ndarray, np.ndarray):
    """
    Load dataset from file: Load, reshape and preprocess data.
    Input:
     - path: Path of the dataset
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - X: input examples [num_examples, 5, 3, H, W]
    - y: golds for the input examples [num_examples]
    """
    try:
        data = np.load(path, allow_pickle=True)["arr_0"]
    except (IOError, ValueError) as err:
        logging.warning(f"[{err}] Error in file: {path}, ignoring the file.")
        return np.array([]), np.array([])
    except:
        logging.warning(f"[Unknown exception, probably corrupted file] Error in file: {path}, ignoring the file.")
        return np.array([]), np.array([])

    X = reshape_x(data)
    y = reshape_y(data)
    return X, y


def load_dataset(path: str, fp: int = 32) -> (np.ndarray, np.ndarray):
    """
    Load dataset from directory: Load, reshape and preprocess data for all the files in a directory.
    Input:
     - path: Path of the directory
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - X: input examples [num_examples_per_file * num_files, 5, 3, H, W]
    - y: golds for the input examples [num_examples_per_file * num_files]
    """
    X: np.ndarray = np.array([])
    y: np.ndarray = np.array([])

    files = glob.glob(os.path.join(path, "*.npz"))
    for file_n, file in enumerate(files):
        print(f"Loading file {file_n + 1} of {len(files)}...")
        X_batch, y_batch = load_file(file, fp)
        if len(X_batch) > 0 and len(y_batch) > 0:
            if len(X) == 0:
                X = X_batch
                y = y_batch
            else:
                X = np.concatenate((X, X_batch), axis=0)
                y = np.concatenate((y, y_batch), axis=0)

    if len(X) == 0 or len(y) == 0:
        # Since this function is used for loading the dev and test set, we want to stop the execution if we don't
        # have a valid test of dev set.
        raise ValueError(f"Empty dataset, all files invalid. Path: {path}")

    return X, y


def train(
        model,
        train_dir: str,
        test_dir: str,
        eval_dir: str,
        output_dir: str,
        batch_size: int,
        num_epoch: int,
        num_load_files_training: int = 10,
        save_checkpoints: bool = True,
        save_best: bool = True,
):
    """
    Train a model
    Input:
    - model: Keras model to train
    - train_dir: Directory where the train files are stored
    - test_dir: Directory where the test files are stored
    - eval_dir: Directory where the eval files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - batch_size: Batch size (Around 10 for 8GB GPU)
    - num_epochs: Number of epochs to do
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set
    Output:
     - float: Accuracy in the development test of the best model
    """

    model.save(os.path.join(output_dir), "model")

    print("Loading test set")
    X_test, y_test = load_dataset(test_dir, fp=32)
    print("Loading eval set")
    X_val, y_val = load_dataset(eval_dir, fp=32)
    print("Loading train set")
    X, y = load_dataset(train_dir, fp=32)

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'mymodel_{epoch}'),
            save_best_only=save_best,
            monitor='val_loss',
            verbose=1)
    ]

    history = model.fit(X, y, batch_size=batch_size, epochs=num_epoch, validation_data=(X_val, y_val), callbacks=callbacks)

    print('\nhistory dict:', history.history)

    print('\n# Evaluate on test data')
    results = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('test loss, test acc:', results)

    return results[1]


def train_new_model(
        num_epoch,
        train_dir,
        test_dir,
        eval_dir,
        output_dir,
        batch_size=4,
        save_checkpoints=True,
        save_best=True,
):
    print("Loading new model")

    max_acc = train(
        model=get_model((270, 480), 5, 2),
        train_dir=train_dir,
        test_dir=test_dir,
        eval_dir=eval_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epoch=num_epoch,
        save_checkpoints=save_checkpoints,
        save_best=save_best,
    )

    print(f"Training finished, max accuracy in the development set {max_acc}")


def continue_training(
        num_epoch,
        checkpoint_path,
        train_dir,
        test_dir,
        eval_dir,
        output_dir,
        batch_size=5,
        save_checkpoints=True,
        save_best=True,
):
    model = load_model(checkpoint_path)

    max_acc = train(
        model=model,
        train_dir=train_dir,
        eval_dir=eval_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epoch=num_epoch,
        save_checkpoints=save_checkpoints,
        save_best=save_best,
    )

    print(f"Training finished, max accuracy in the development set {max_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train_new", action="store_true", help="Train a new model",
    )

    group.add_argument(
        "--continue_training",
        action="store_true",
        help="Restore a checkpoint and continue training",
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing the train files",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing the test files",
    )

    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing the eval files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the model and checkpoints are going to be saved",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch size for training (10 for a 8GB GPU seems fine)",
    )

    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of epochs to perform",
    )

    parser.add_argument(
        "--not_save_checkpoints",
        action="store_false",
        help="Do NOT save a checkpoint each epoch (Each checkpoint will rewrite the previous one)",
    )

    parser.add_argument(
        "--not_save_best",
        action="store_false",
        help="Dot NOT save the best model in the development set",
    )

    args = parser.parse_args()

    if args.train_new:
        train_new_model(
            train_dir=args.train_dir,
            eval_dir=args.eval_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epoch=args.num_epochs,
            save_checkpoints=args.not_save_checkpoints,
            save_best=args.not_save_best,
        )
