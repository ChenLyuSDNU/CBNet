class DefaultConfigs(object):
    # 1.string parameters
    train_data = "./data/new/"
    test_data = "./data/new/test/"
    val_data = "no"
    model_name = "shuffle1"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "1"

    # 2.numeric parameters
    epochs = 1
    batch_size = 2
    img_height = 224
    img_weight = 224
    num_classes = 5
    num_classes_1 = 3
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4


config = DefaultConfigs()
