from src import get_dataset_ner


if __name__ == '__main__':

    data_split_all, label_to_id, language = get_dataset_ner('ontonote5')
    print(label_to_id, language)
