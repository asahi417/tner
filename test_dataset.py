from src import get_dataset_ner

VALID_DATASET = ['conll_2003', 'wnut_17', 'ontonote5', 'mit_movie_trivia', 'mit_restaurant']

if __name__ == '__main__':
    for t in VALID_DATASET:
        get_dataset_ner(t)
