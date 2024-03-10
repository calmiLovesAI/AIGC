def load_pretrained_state_dict(model, pretrained_state_dict):
    model_dict = model.state_dict()
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_state_dict.items():
        if k in model_dict.keys() and model_dict[k].shape == v.shape:
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print(f"Successfully loaded {len(load_key)} keys, they are: {load_key[:20]}...")
    print(f"Failed to load {len(no_load_key)} keys, they are: {no_load_key[:20]}...")
