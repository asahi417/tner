from src import get_dataset_ner


if __name__ == '__main__':

    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('ontonote5')
    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('conll_2003')
    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('mit_movie_trivia')
    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('mit_restaurant')
    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('ner-cogent-en')
    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('ner-cogent-ja-mecab')
    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('panx_dataset/en')
    data_split_all, label_to_id, language, unseen_entity_set = get_dataset_ner('panx_dataset/ja')
