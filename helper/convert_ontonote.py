import jsonlines

test = './cache/OntoNotes/test.json'
train = './cache/OntoNotes/train.json'

with jsonlines(test) as reader:
    for obj in reader:
        obj['tokens']