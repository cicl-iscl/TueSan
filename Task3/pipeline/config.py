from ray import tune

train_config = {
    # --- hyperparameters ----
    "lr": 0.05,
    "momentum": 0,
    "nesterov": False,
    "weight_decay": 0,
    "batch_size": 64,
    "epochs": 10,
    "embedding_dim": 64,
    "hidden_dim": 256,
    "max_ngram": 8,
    "dropout": 0.0,
    "char2token_mode": "rnn",
    # ------------------------
    "name": "test_translit",
    "translit": True,
    "test": True,
    "cuda": True,
    "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
    "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
    "test_path": "/data/jingwen/sanskrit/test_set",
    "out_folder": "../sanskrit",
    "submission_dir": "result_submission",
    "checkpoint_dir": "./checkpoint",
}


tune_config = config = {
    # --- hyperparameters ----
    "lr": 0.05,
    "momentum": 0,
    "nesterov": False,
    "weight_decay": 0,
    # "batch_size": tune.choice([16, 32, 64, 128]),
    "batch_size": 64,
    "epochs": tune.choice([1, 2]),
    "embedding_dim": tune.choice([32, 64, 128, 256]),
    "hidden_dim": 256,
    "max_ngram": 8,
    "dropout": 0,
    "char2token_mode": "rnn",
    # ------------------------
    "name": "test_translit",
    "translit": True,
    "test": True,
    "cuda": True,
    "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
    "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
    "test_path": "/data/jingwen/sanskrit/test_set",
    "out_folder": "../sanskrit",
    "submission_dir": "result_submission",
    "checkpoint_dir": "./checkpoint",
}
