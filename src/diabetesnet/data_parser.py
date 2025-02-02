from pathlib import Path

import pandas as pd
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset, random_split


class DataParser:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

    @staticmethod
    def parse_data(data_dir):
        if Path(data_dir).suffix == ".csv":
            data = pd.read_csv(data_dir)
        else:
            assert False, "Only CSV files are supported"

        target = data["Outcome"]
        features = data[data.columns.difference(["Outcome", "Pregnancies"])]
        return features, target

    def get_data_loader(self):
        features, target = self.parse_data(self.data_dir)
        features = tensor(features.values).float()
        target = tensor(target.values).float()

        dataset = TensorDataset(features, target)

        # split the data into train and test
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        data_loader_train = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        data_loader_test = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        print(f"Train data loader: {len(data_loader_train)}")
        print(f"Test data loader: {len(data_loader_test)}")

        return data_loader_train, data_loader_test
