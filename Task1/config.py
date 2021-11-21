from ray import tune

train_config = {
    "lr": 0.05,
    "momentum": 0,
    "nesterov": False,
    "weight_decay": 0,
    "name": "trained_translit",
    "translit": True,
    "test": True,
    "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
    "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
    "test_path": "/data/jingwen/sanskrit/test_set",
    "batch_size": 16,
    "epochs": 20,
    "embedding_dim": 128,
    "hidden_dim": 512,
    "max_ngram": 8,
    "dropout": 0.1,
    "use_lstm": True,
    "cuda": True,
    "out_folder": "../sanskrit",
    "submission_dir": "result_submission",
    "checkpoint_dir": "./checkpoint",
}


tune_config = config = {
    "lr": 0.05,
    "batch_size": tune.choice([16, 64]),
    # "batch_size": 64,
    "epochs": tune.choice([15, 20, 25]),
    "momentum": 0,
    "nesterov": False,
    "weight_decay": 0,
    "max_ngram": 8,
    "dropout": tune.choice([0.0, 0.1]),
    "use_lstm": tune.choice([True, False]),
    "hidden_dim": tune.choice([128, 256, 512]),
    "embedding_dim": tune.choice([32, 64, 128]),
    "name": "best_translit",
    "translit": tune.choice([True, False]),
    "test": True,
    "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
    "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
    "test_path": "/data/jingwen/sanskrit/test_set",
    "cuda": True,
    "out_folder": "../sanskrit",
    "submission_dir": "result_submission",
    "checkpoint_dir": "./checkpoint",
}
