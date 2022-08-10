import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib import cm, colors  # for colormap https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
from tner import TransformersNER

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

NER_MODEL = os.getenv('NER_MODEL', 'tner/roberta-large-wnut2017')
DEBUG = False
DUMMY = {
   'prediction': [['B-person', 'I-person', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-location']],
   'probability': [[0.9967652559280396, 0.9994561076164246, 0.9986955523490906, 0.9947081804275513, 0.6129112243652344, 0.9984312653541565, 0.9868122935295105, 0.9983410835266113, 0.9995284080505371, 0.9838910698890686]],
   'input': [['Jacob', 'Collier', 'is', 'a', 'Grammy', 'awarded', 'English', 'artist', 'from', 'London']],
   'entity_prediction': [[
       {'type': 'person', 'entity': ['Jacob', 'Collier'], 'position': [0, 1], 'probability': [0.9967652559280396, 0.9994561076164246]},
       {'type': 'location', 'entity': ['London'], 'position': [9], 'probability': [0.9838910698890686]}
    ]]
}
if not DEBUG:
    MODEL = TransformersNER(NER_MODEL)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_ckpt": NER_MODEL})


@app.post("/process")
async def process(request: Request):
    data_json = await request.json()
    input_text = data_json['input_text']
    max_len = int(data_json['max_len'])
    if DEBUG:
        ner_result = DUMMY
    else:
        ner_result = MODEL.predict([input_text], max_length=max_len)
    ner_result['html'] = generate_html(ner_result)
    return ner_result


def generate_html(ner_result, separator=' '):
    """ add <mark> tag to the entity """
    token = ner_result['input'][0]
    entity = ner_result['entity_prediction'][0]
    html = '<p class="bold"> Input sentence: </p>'
    html_entity = '<p class="bold"> Entities: </p>'
    last_end = 0

    # generate color map on the fly for better color pattern
    unique_type = list(set([i['type'] for i in entity]))
    color_map = cm.Dark2(range(len(unique_type)))
    color_bar = {t: colors.rgb2hex(c) for c, t in zip(color_map, unique_type)}
    content_list = []
    for n, ent in enumerate(entity):
        _type = ent['type']
        content_list.append(separator.join(token[last_end:ent["position"][0]]))
        content_list.append(
            f'<span style="background:{color_bar[_type]};color:white;">{separator.join([token[i] for i in ent["position"]])}</span>'
        )
        last_end = ent["position"][-1] + 1
        html_entity += f'* {n+1}. {separator.join(ent["entity"])}: <span style="font-weight:bold;color:{color_bar[_type]};">{_type}</span> <br>'
    content_list.append(separator.join(token[last_end:]))
    html += separator.join(content_list)
    html += '<br><br>'
    html += html_entity
    return html
