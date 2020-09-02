from src import get_dataset_ner


if __name__ == '__main__':

    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('mit_restaurant')
    print(label_to_id, unseen_entity_set)
    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('mit_movie_trivia', label_to_id=label_to_id)
    print(label_to_id, unseen_entity_set)
