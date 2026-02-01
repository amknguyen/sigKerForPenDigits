import numpy as np
import pandas as pd
import iisignature as iisig
from typing import Optional
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_data():
    base_dir = Path.cwd().parent
    data_dir = base_dir / "data"
    training_dir = data_dir / "pendigits.tra"
    test_dir = data_dir / "pendigits.tes"

    training = pd.read_csv(training_dir, header=None)
    training_x = training.iloc[:, :16].values

    training_x_paths = training_x.reshape(-1, 8, 2)
    training_y = training.iloc[:, 16].values

    test = pd.read_csv(test_dir, header=None)
    test_x = test.iloc[:, :16].values

    test_x_paths = test_x.reshape(-1, 8, 2)
    test_y = test.iloc[:, 16].values

    return training_x_paths, training_y, test_x_paths, test_y

def calc_signature_data(paths, level = 2, log: Optional = None):
    if log is None:
        return np.array([iisig.sig(path, level) for path in paths])
    else:
        return np.array([iisig.logsig(path, log) for path in paths])

def time_augmentation(paths):
    n_samples, x, y = paths.shape
    t = np.linspace(0, 1, 8)
    t_full = np.tile(t, (n_samples, 1)).reshape(n_samples, 8, 1)

    return np.concatenate((t_full, paths), axis=2)

def main(level = 2, n_estimators = 100):
    training_x_paths, training_y, test_x_paths, test_y = load_data()

    training_x_paths = time_augmentation(training_x_paths)
    test_x_paths = time_augmentation(test_x_paths)

    training_sig = calc_signature_data(training_x_paths, level = level)
    test_sig = calc_signature_data(test_x_paths, level = level)

    scaler = StandardScaler()
    training_sig = scaler.fit_transform(training_sig)
    test_sig = scaler.transform(test_sig)

    forest = RandomForestClassifier(n_estimators=n_estimators)
    forest.fit(training_sig, training_y)

    print(forest.score(test_sig, test_y))

    return training_x_paths, training_y, test_x_paths, test_y, training_sig, test_sig, forest

if __name__ == "__main__":
    main()