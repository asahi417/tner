import os
import argparse

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib import cm, colors  # for colormap https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
from tner import TransformersNER


def main():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('--path-static', help='path to static folder', default='static', type=str)
    parser.add_argument('--path-template', help='path to template folder', default='template', type=str)
    parser.add_argument('--ner-model', help='path or huggingface alias of ner model',
                        default='asahi417/tner-xlm-roberta-large-ontonotes5', type=str)
    args = parser.parse_args()

    app = FastAPI()
    assert os.path.isdir(args.path_static), f"{args.path_static} is not directly"
    assert os.path.isdir(args.path_template), f"{args.path_template} is not directly"
    app.mount("/static", StaticFiles(directory=args.static), name="static")
    templates = Jinja2Templates(directory=args.path_template)

    model = TransformersNER(args.ner_model)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request, "model_ckpt": args.ner_model})

    @app.post("/process")
    async def process(request: Request):
        data_json = await request.json()
        input_text = data_json['input_text']
        max_len = int(data_json['max_len'])
        ner_result = model.predict([input_text], max_seq_length=max_len)[0]
        ner_result['html'] = generate_html(ner_result)
        return ner_result

    def generate_html(ner_result):
        """ add <mark> tag to the entity """
        sentence = ner_result['sentence']
        entity = ner_result['entity']
        html = '<p class="bold"> Input sentence: </p>'
        html_entity = '<p class="bold"> Entities: </p>'
        last_end = 0

        # generate color map on the fly for better color pattern
        unique_type = list(set([i['type'] for i in entity]))
        color_map = cm.Dark2(range(len(unique_type)))
        color_bar = {t: colors.rgb2hex(c) for c, t in zip(color_map, unique_type)}

        for n, ent in enumerate(entity):
            s, e = ent['position']
            mention = ent['mention']
            _type = ent['type']
            html += sentence[last_end:s]
            html += '<span style="background:{0};color:white;">{1}</span>'.format(color_bar[_type], sentence[s:e])
            last_end = e
            html_entity += '* {0}. {1}: <span style="font-weight:bold;color:{2};">{3}</span> <br>'.format(
                n + 1, mention, color_bar[_type], _type)

        html += sentence[last_end:]
        html += '<br><br>'
        html += html_entity
        return html
