from pathlib import Path
from torch import nn, optim, no_grad, Tensor, cat, save, device

from torch.utils.data import DataLoader
from torch.cuda import is_available as is_cuda_available


class DiabetesNet(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_classes: int, dropout=0.3
    ):
        super(DiabetesNet, self).__init__()
        self.device = device("cuda" if is_cuda_available() else "cpu")
        self.layers = self.model(input_size, hidden_size, num_classes, dropout)
        self.to(self.device)

    def model(
        self, input_size: int, hidden_size: int, num_classes: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),  # input layer
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),  # hidden layer 1
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2),  # hidden layer 2
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),  # output layer
            nn.Sigmoid(),
        )

    def train_model(
        self,
        dataloader: DataLoader,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ) -> None:

        if not isinstance(dataloader, DataLoader):
            assert False, "DataLoader object required"

        optimizer = optim.Adam(
            lr=learning_rate, params=self.parameters(), weight_decay=weight_decay
        )
        criterion = nn.BCELoss()
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels.view(-1, 1).float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader)}")

        print(f"Epoch [{epochs}/{epochs}], Loss: {total_loss / len(dataloader)}")

    def evaluate(self, data_loader: DataLoader, threshold=0.5) -> float:
        self.eval()
        labels_predicted = []
        labels_actual = []
        for batch in data_loader:
            inputs, outcome = batch
            inputs = inputs.to(self.device)
            outcome = outcome.to(self.device)
            with no_grad():
                pred = (self(inputs) >= threshold).float()

            labels_predicted.append(pred.view(-1))
            labels_actual.append(outcome.view(-1))

        labels_predicted = cat(labels_predicted)
        labels_actual = cat(labels_actual)

        print("Predicted: ", labels_predicted)
        print("Actual: ", labels_actual)

        accuracy = float((labels_predicted == labels_actual).sum() / len(labels_actual))
        print("Accuracy", accuracy)
        return accuracy

    def forward(self, x) -> Tensor:
        x = x.to(self.device)
        return self.layers(x)

    def save_model(self, path: Path):
        save(self.state_dict(), path)
