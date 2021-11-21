from ray import tune

train_config = {
    "lr": 0.05,
    "momentum": 0,
    "nesterov": False,
    "weight_decay": 0.0,
    "name": "test_translit",
    "translit": True,
    "test": True,
    "train_path": "/data/jingwen/sanskrit/wsmp_train.json",
    "eval_path": "/data/jingwen/sanskrit/corrected_wsmp_dev.json",
    "test_path": "/data/jingwen/sanskrit/test_set",
    "batch_size": 64,
    "epochs": 1,
    "embedding_size": 64,
    "encoder_hidden_size": 64,
    "encoder_max_ngram": 6,
    "encoder_char2token_mode": "rnn",
    "classifier_hidden_dim": 64,
    "classifer_num_layers": 2,
    "dropout": 0.05,
    "tag_rules": True,
    "stemming_rule_cutoff": 5,
    "cuda": True,
    "out_folder": "../sanskrit",
    "submission_dir": "result_submission",
    "checkpoint_dir": "./checkpoint",
}


tune_config = config = {
    "lr": 0.05,
    # "batch_size": tune.choice([16, 32, 64, 128]),
    "batch_size": 64,
    "epochs": tune.choice([15, 20]),
    "momentum": 0,
    "nesterov": False,
    "weight_decay": 0,
    "embedding_size": tune.choice([32, 64, 128]),
    "encoder_hidden_size": tune.choice([256, 512]),
    "encoder_max_ngram": 6,
    "encoder_char2token_mode": tune.choice(["max", "rnn"]),
    "classifier_hidden_dim": tune.choice([256, 512]),
    "classifer_num_layers": 2,
    "dropout": 0.0,
    "tag_rules": tune.choice([True, False]),
    "stemming_rule_cutoff": tune.choice([1, 5, 50]),
    "name": "test_translit",
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