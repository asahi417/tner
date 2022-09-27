# Dataset Card
NER dataset shared on the [huggingface TNER organization](https://huggingface.co/tner).

| Dataset           | Alias (link)                                                                      | Domain                                  | Size (train/valid/test) | Language | Entity Size |
|-------------------|-----------------------------------------------------------------------------------|-----------------------------------------|-------------------------|----------|-------------|
| Ontonotes5        | [`tner/ontonotes5`](https://huggingface.co/datasets/tner/ontonotes5)              | News, Blog, Dialogue                    | 59,924/8,528/8,262      | en       | 18          |
| CoNLL2003         | [`tner/conll2003`](https://huggingface.co/datasets/tner/conll2003)                | News                                    | 14,041/3,250/3,453      | en       | 4           |
| BioNLP2004        | [`tner/bionlp2004`](https://huggingface.co/datasets/tner/bionlp2004)              | Biochemical                             | 16,619/1,927/3,856      | en       | 5           |
| BioCreative V CDR | [`tner/bc5cdr`](https://huggingface.co/datasets/tner/bc5cdr)                      | Biomedical                              | 5,228/5,330/5,865       | en       | 2           |
| FIN               | [`tner/fin`](https://huggingface.co/datasets/tner/fin)                            | Financial News                          | 1,014/303/150           | en       | 4           |
| MIT Movie         | [`tner/mit_movie_trivia` ](https://huggingface.co/datasets/tner/mit_movie_trivia) | Movie Review                            | 6,816/1,000/1,953       | en       | 12          |
| MIT Restaurant    | [`tner/mit_restaurant`](https://huggingface.co/datasets/tner/mit_restaurant)      | Restaurant Review                       | 6,900/760/1,521         | en       | 8           |
| WNUT2017          | [`tner/wnut2017`](https://huggingface.co/datasets/tner/wnut2017)                  | Twitter, Reddit, StackExchange, YouTube | 2,395/1,009/1,287       | en       | 6           |
| BTC               | [`tner/btc`](https://huggingface.co/datasets/tner/btc)                            | Twitter                                 | 1,014/303/150           | en       | 3           |
| Tweebank NER      | [`tner/tweebank_ner`](https://huggingface.co/datasets/tner/tweebank_ner)          | Twitter                                 | 1,639/710/1,201         | en       | 4           |
| TTC               | [`tner/ttc`](https://huggingface.co/datasets/tner/ttc), [`tner/ttc_dummy`](https://huggingface.co/datasets/tner/ttc_dummy)| Twitter | 9,995/500/1,477 | en       | 3           |
| TweetNER7         | [`tner/tweetner7`](https://huggingface.co/datasets/tner/tweetner7)                | Twitter                                 | 7,111/576/2,807 (*see the dataset page)        | en       | 7           |

Multilingual dataset follows below.
| Dataset           | Alias (link)                                                                      | Domain                                  | Language | Entity Size |
|-------------------|-----------------------------------------------------------------------------------|-----------------------------------------|-------------------------|----------|
| WikiANN (Panx)         | [`tner/wikiann`](https://huggingface.co/datasets/tner/wikiann)                | Wikipedia                                 | 160+       | 3           |
| WikiNeural         | [`tner/wikineural`](https://huggingface.co/datasets/tner/wikineural)                | Wikipedia                                 | 9       | 16           |
| MultiNERD         | [`tner/multinerd`](https://huggingface.co/datasets/tner/multinerd)                | Wikipedia, WikiNews                                 | 16       | 18           |
