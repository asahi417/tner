import argparse
import json
import torch
import transformers
import os
from glob import glob


def get_options():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('-c', '--checkpoint-dir', help='checkpoint to load', required=True, type=str)
    return parser.parse_args()


def convert(checkpoint: str):
    with open('{}/parameter.json'.format(checkpoint), 'r') as f:
        config = json.load(f)
    with open('{}/label_to_id.json'.format(checkpoint), 'r') as f:
        label_to_id = json.load(f)

    id_to_label = {v: str(k) for k, v in label_to_id.items()}
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        config['transformers_model'],
        config=transformers.AutoConfig.from_pretrained(
            config['transformers_model'],
            num_labels=len(id_to_label),
            id2label=id_to_label,
            label2id=label_to_id,
            cache_dir='./cache')
    )
    checkpoint_file = '{}/model.pt'.format(checkpoint)
    stats = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(stats['model_state_dict'])
    model.save_pretrained(checkpoint)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['transformers_model'], cache_dir='./cache')
    tokenizer.save_pretrained(checkpoint)

    os.remove('{}/model.pt'.format(checkpoint))
    os.remove('{}/label_to_id.json'.format(checkpoint))


if __name__ == '__main__':
    opt = get_options()
    if opt.checkpoint_dir is not None:
        convert(opt.checkpoint_dir)
    else:
        for i in glob('./ckpt/*/*'):
            print('converting: {}'.format(i))
            convert(i)

