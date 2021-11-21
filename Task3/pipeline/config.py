from ray import tune

train_config = {
    # --- hyperparameters ----
    "lr": 0.05,
    "momentum": 0,
    "nesterov": False,
    "weight_decay": 0,
    "batch_size": 64,
    "epochs": 1,
    "embedding_dim": 64,
    "hidden_dim": 64,
    "max_ngram": 4,
    "dropout": 0.0,
    "char2token_mode": "max",
    # ------------------------
    "name": "train_translit",
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
    "batch_size": tune.choice([16, 64]),
    "epochs": 15,
    "embedding_dim": 128,
    "hidden_dim": tune.choice([256, 512]),
    "max_ngram": 8,
    "dropout": tune.choice([0.0, 0.1]),
    "char2token_mode": tune.choice(["rnn", "max"]),
    # ------------------------
    "name": "best_translit",
    "translit": tune.choice([True, False]),
    "test": True,
    "cuda": True,
    "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
    "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
    "test_path": "/data/jingwen/sanskrit/test_set",
    "out_folder": "../sanskrit",
    "submission_dir": "result_submission",
    "checkpoint_dir": "./checkpoint",
}
