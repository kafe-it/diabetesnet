from diabetesnet.data_parser import DataParser
from diabetesnet.neural_network import DiabetesNet
from diabetesnet.utils.argparse import Arguments, create_parser
from datetime import datetime


def hyperparameter_tuning_mode(args: Arguments) -> dict:
    import optuna

    def objective(trial):
        train_loader, test_loader = prepare_data_loader(args)

        input_size = args.input_size
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        num_classes = args.num_classes
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        epochs = args.epochs
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)

        model = DiabetesNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
        )
        model.train_model(
            train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        accuracy = model.evaluate(test_loader)
        return accuracy

    print("Hyperparameter tuning started")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print("Hyperparameter tuning successful")
    print("Best hyperparameters: ", study.best_params)
    return study.best_params


def default_mode(args: Arguments):
    train_loader, test_loader = prepare_data_loader(args)

    print("Neural network training started")
    model = DiabetesNet(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
        dropout=args.dropout,
    )
    model.train_model(
        train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print("Neural network training successful")

    print("Model evaluation started")
    model.evaluate(test_loader)
    print("Model evaluation successful")

    print("Model saving started")
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    model.save_model(f"{args.output}/model_{time}.pth")
    print("Model saving successful")


def prepare_data_loader(args: Arguments):
    print("Data parsing started")
    data_parser = DataParser(data_dir=args.input, batch_size=args.batch_size)
    train_loader, test_loader = data_parser.get_data_loader()
    print("Data parsing successful")
    return train_loader, test_loader


if __name__ == "__main__":
    args = create_parser()
    if args.hyperparam_tuning:
        params = hyperparameter_tuning_mode(args)
        default_mode(Arguments(params))
    else:
        default_mode(args)
