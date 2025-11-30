from models.simple_models import (
    DeepESN,
    DeepRNN,
    SimpleESN,
    SimpleESNWithNonLinearReadout,
    SimpleRNN,
)


def create_model_from_config(config: dict, input_size: int, num_classes: int):
    model_type = config["type"]

    if model_type == "RNN":
        model = SimpleRNN(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_classes=num_classes,
            num_layers=config["num_layers"],
        )
        name = f"RNN_h{config['hidden_size']}_L{config['num_layers']}"

    elif model_type == "DeepRNN":
        model = DeepRNN(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=num_classes,
        )
        name = f"DeepRNN_h{config['hidden_size']}_L{config['num_layers']}"

    elif model_type == "ESN":
        model = SimpleESN(
            input_size=input_size,
            reservoir_size=config["reservoir_size"],
            num_classes=num_classes,
        )
        name = f"ESN_r{config['reservoir_size']}"

    elif model_type == "ESNNonLinear":
        hidden_readout_size = config.get("hidden_readout_size", 256)
        model = SimpleESNWithNonLinearReadout(
            input_size=input_size,
            reservoir_size=config["reservoir_size"],
            num_classes=num_classes,
            hidden_readout_size=hidden_readout_size,
        )
        name = f"ESNNonLinear_r{config['reservoir_size']}_h{hidden_readout_size}"

    elif model_type == "DeepESN":
        model = DeepESN(
            input_size=input_size,
            reservoir_size=config["reservoir_size"],
            num_layers=config["num_layers"],
            num_classes=num_classes,
        )
        name = f"DeepESN_r{config['reservoir_size']}_L{config['num_layers']}"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, name
