import torch


def dynamic_load_weights_pt(model: torch.nn.Module, weights: dict):
    """
    Dynamically loads weights into a PyTorch model, handling potential mismatches in layer sizes.
    Provides verbose feedback on any mismatched layers and skips only those layers while keeping
    the rest intact.

    :param model: The PyTorch model into which the weights are being loaded.
    :param weights: A dictionary containing the model weights, typically loaded from a checkpoint or pre-trained model.
    """
    # Get the current model's state dictionary
    current_model_dict = model.state_dict()

    # Track mismatched layers
    mismatched_layers = []

    # Dry run: attempt loading weights with strict=True to identify mismatches
    try:
        model.load_state_dict(weights, strict=True)
        print("All weights successfully loaded with strict=True.")
        return
    except RuntimeError as e:
        print("Identifying mismatched layers...")
        error_message = str(e)
        lines = error_message.split('\n')
        for line in lines:
            if "size mismatch" in line:
                mismatched_layer = line.split("size mismatch for ")[1].split(":")[0]
                mismatched_layers.append(mismatched_layer)

    # Log mismatched layers
    if mismatched_layers:
        print("The following layers have size mismatches and will be skipped:")
        for layer in mismatched_layers:
            print(f"- {layer}")

    # Remove mismatched layers from weights and reload
    filtered_weights = {
        k: v if k not in mismatched_layers else current_model_dict[k]
        for k, v in weights.items()
    }
    model.load_state_dict(filtered_weights, strict=False)
    print("Model weights loaded with mismatched layers resolved.")