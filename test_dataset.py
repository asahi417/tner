from src import get_dataset_ner

VALID_DATASET = ['conll_2003', 'wnut_17', 'ontonote5', 'mit_movie_trivia', 'mit_restaurant']

if __name__ == '__main__':
    a = get_dataset_ner('conll_2003', lower_case=True)
    print(a[0]['train']['data'][0])
    # for t in VALID_DATASET:
    #     get_dataset_ner(t)
