import argparse


class Arguments:
    def __init__(self, args):
        # Data Paths
        self.input = args.input
        self.output = args.output

        # Neural Network Hyperparameters
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        # Optuna Hyperparameter Tuning
        self.hyperparam_tuning = args.hyperparam_tuning


def create_parser() -> Arguments:
    parser = argparse.ArgumentParser(description="Diabetes Network")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input file (csv format)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output folder",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=7,
        help="Input size for the neural network",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden size for the neural network",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Number of classes for the neural network",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout for the neural network",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0015,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for training",
    )
    parser.add_argument(
        "--hyperparam-tuning",
        action="store_true",
        help="Enable hyperparameter tuning",
    )
    return Arguments(parser.parse_args())
