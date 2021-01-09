import os
from glob import glob

ckpt = './ckpt'
for i in glob('{}/*/*/*_ignore.json'):
    os.rename(i, i.replace('_ignore', '_span'))

# for i in glob('{}/*/*'):
#     data_name = i.split('/')[-1]
#     for _i in glob('{}/test*.json'.format(i)):
#         if data_name in _i:
#             continue
#         if 'all_english' in _i:
#             continue
#
#         if '_span' not in _i:
#             os.remove(_i)


