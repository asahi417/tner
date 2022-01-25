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
    if i.endswith('format'):
        continue

    total = ''
    with open(i) as f:
        tmp = f.read()
        for _tmp in tmp.split('\n\n'):
            start = '#\n'
            for t in _tmp.split('\n'):
                tag = t.split(' ')[-1]
                for k, v in tag_map.items():
                    tag = tag.replace(k, v)
                start = start + tag + '\n'
            start = start + '\n'
            total += start

    with open(i+'.format', 'w') as f:
        f.write(total)


