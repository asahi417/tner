import logging
import tner
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

trainer = tner.Trainer(model='albert-base-v1', checkpoint_dir='./tmp', dataset=['./cache/custom_dataset'], epoch=1, batch_size=1)
trainer.train()
