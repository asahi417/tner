import logging
from tner import TransformersNER, get_dataset
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

model = TransformersNER(model='ckpt', max_length=128)
print(model.tokenizer.sp_token_start, model.tokenizer.sp_token_end)
print(model.tokenizer.prefix)
print(model.label2id)
data, _, _, _ = get_dataset('conll2003', label_to_id=model.label2id)
model.span_f1(data['test']['data'], labels=data['test']['label'], batch_size=2)
# loader = model.get_data_loader(inputs=data['test']['data'], labels=data['test']['label'])
# # for i in loader:
# #     print(i)
# #     input()
#
# for label, data in zip(data['test']['label'], data['test']['data']):
#     print(label)
#     print(data)
#     pred, tokens = model.predict([data])
#     print(list(zip(pred[0], tokens[0])))
#
#     # pred = model.predict([data], decode_bio=True)
#     # input(pred)
#     input()
