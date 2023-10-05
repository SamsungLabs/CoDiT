def to_device(nested_tensors, device):
    new_nested_tensors = dict()
    for key, value in nested_tensors.items():
        new_nested_tensors.update({key: value.to(device)})
    return new_nested_tensors
