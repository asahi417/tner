from tner import TrainTransformersNER
trainer = TrainTransformersNER(
        checkpoint_dir='./ckpt',
        dataset='/Users/asahi/Desktop/data_seyyaw',
        transformers_model='xlm-roberta-base',
        random_seed=1234,
        lr=1e-5,
        total_step=10,
        warmup_step=10,
        batch_size=1,
        max_seq_length=128)
trainer.train()
