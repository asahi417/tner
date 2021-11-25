from tner import TransformersNER, get_dataset

lm = TransformersNER('ckpt/epoch_10', max_length=128)
dataset_split, label_to_id, language, unseen_entity_set = get_dataset('wnut2017', lower_case=False)
input(dataset_split.keys())
data = dataset_split['test']['data'][:10]
label = dataset_split['test']['label'][:10]
# print(label)
metric = lm.span_f1(data, label, batch_size=16)
print(metric)
pred = lm.predict(data, batch_size=16, decode_bio=True)
print(pred)
