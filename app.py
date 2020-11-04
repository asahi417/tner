import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib import cm, colors  # for colormap https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
from tner import TransformerNER

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_CKPT = os.getenv('MODEL_CKPT', './ckpt/default')
if MODEL_CKPT == '':
    MODEL = None
else:
    MODEL = TransformerNER(checkpoint=MODEL_CKPT)
DUMMY = {
    'sentence': 'Jacob Collier lives in London',
    'entity': [
        {'mention': 'Jacob Collier', 'position': [0, 13], 'probability': 0.8, 'type': 'person'},
        {'mention': 'London', 'position': [23, 29], 'probability': 0.8, 'type': 'location'}
    ]}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_ckpt": MODEL_CKPT})


@app.post("/process")
async def process(request: Request):
    data_json = await request.json()
    input_text = data_json['input_text']
    max_len = int(data_json['max_len'])
    if MODEL is None:
        ner_result = DUMMY
    else:
        ner_result = MODEL.predict([input_text], max_seq_length=max_len)[0]
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
