import os
from glob import glob

ckpt = './ckpt'
for i in glob('{}/*/*/*_ignore*.json'.format(ckpt)):
    os.rename(i, i.replace('_ignore', '_span'))

# for i in glob('{}/*/*'.format(ckpt)):
#     if not os.path.isdir(i):
#         continue
#     data_name = i.split('/')[-1]
#     is_lower = '_lower' in i
#     print(i, is_lower, data_name)
#     input()
#     for _i in glob('{}/test*.json'.format(i)):
#         if data_name in _i:
#             continue
#         if 'all_english' in _i:
#             continue
#
#         if '_span' not in _i:
#             os.remove(_i)
#
#         if '_lower' not in _i and is_lower:
#             os.remove(_i)
#
#         if '_lower' in _i and not is_lower:
#             os.remove(_i)


