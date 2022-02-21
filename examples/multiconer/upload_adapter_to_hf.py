

from transformers import AutoModelWithHeads


def upload_hf(base_model_name, adapter_name, repo):
	model = AutoModelWithHeads.from_pretrained(base_model_name)
	adapter_name = model.load_adapter(adapter_name, source="hf")
	model.active_adapters = adapter_name
	model.push_adapter_to_hub(
		repo,
		"ner",
		adapterhub_tag="named-entity-recognition/multiconer",
		datasets_tag="multiconer"
	)
	model.load_adapter("asahi417/{}".format(repo), source="hf")

upload_hf('xlm-roberta-large', 'xlm_roberta_large_multi_adapter', 'tner-xlm-roberta-large-multiconer-multi-adapter')
upload_hf('xlm-roberta-large', 'xlm_roberta_large_mix_adapter', 'tner-xlm-roberta-large-multiconer-mix-adapter')
upload_hf('roberta-large', 'roberta_large_en_adapter', 'tner-roberta-large-multiconer-en-adapter')


