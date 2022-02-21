import os
from glob import glob

tag_map = {
    'work-of-art': 'CW',
    'person': 'PER',
    'corporation': 'CORP',
    'group': 'GRP',
    'location': 'LOC',
    'product': 'PROD',
}


for i in glob('cache/multiconer_test/*.prediction.*'):
    term = i.split('/')[-1].split('.prediction')[0]
    with open('cache/multiconer_test/{}.conll'.format(term)) as f:
        anchor_data = f.read()

    if i.endswith('format'):
        continue
    print(i)
    total = ''
    with open(i) as f:
        tmp = f.read()
        tmp_split = [i for i in tmp.split('\n\n') if len(i) > 0]
        anchor_data_split = [i for i in anchor_data.split('\n\n') if len(i) > 0]
        assert len(tmp_split) >= len(anchor_data_split)
        for _tmp, _tmp_anchor in zip(tmp_split[:len(anchor_data_split)], anchor_data_split):
            # start = '#\n'
            start = ''

            _tmp_split = [i for i in _tmp.split('\n') if len(i) > 0]
            _tmp_anchor = [i for i in _tmp_anchor.split('\n')[1:] if len(i) > 0]

            if len(_tmp_split) != len(_tmp_anchor):
                assert len(_tmp_split) < len(_tmp_anchor), '{}\n{}'.format(_tmp_split, _tmp_anchor)
                _new_tmp_split = []
                for n in range(len(_tmp_anchor)):
                    if 0 == len(_tmp_split):
                        _new_tmp_split.append('# O')
                        break
                    if _tmp_anchor[n].startswith('#') or _tmp_anchor[n].startswith('_______') or _tmp_anchor[n].split(' ')[-1] == '_':
                        _new_tmp_split.append('# O')
                    else:
                        _new_tmp_split.append(_tmp_split.pop(0))
                _tmp_split = _new_tmp_split
                if len(_tmp_split) != len(_tmp_anchor):
                    for u in zip(_tmp_split, _tmp_anchor):
                        print(u)
                    print(_tmp_split, len(_tmp_split))
                    print(_tmp_anchor, len(_tmp_anchor))
                    input()

            for t, a in zip(_tmp_split, _tmp_anchor):

                tag = t.split(' ')[-1]
                for k, v in tag_map.items():
                    tag = tag.replace(k, v)
                start = start + tag + '\n'
            start = start + '\n'
            total += start

    os.makedirs(i+'.format', exist_ok=True)

    with open('{}.format/{}.pred.conll'.format(i, term.replace('_test', '')), 'w') as f:
        f.write(total)


