import numpy as np
import options.options as option
import utils.util as util
from models import create_model
from PIL import Image
import torch
import streamlit as st
from liif.utils import make_coord
import liif.models as liif_models


if 'model' not in st.session_state:
    # our
    opt = option.parse('webui.json', is_train=False)
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    model.netG.eval()
    st.session_state['model'] = model

    # nESRGAN+
    opt = option.parse('nESRGANplus.json', is_train=False)
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
    opt = option.dict_to_nonedict(opt)
    nESRGANplus = create_model(opt)
    nESRGANplus.netG.eval()
    st.session_state['nESRGANplus'] = nESRGANplus

    # LIIF
    st.session_state['LIIF'] = liif_models.make(torch.load('LIIF_model.pth', map_location=torch.device('cpu'))['model'], load_sd=True)


def sr(upload):
    im = Image.open(upload)
    col1.write("Original Image :camera:")

    w, h = im.size
    width = 200
    if (w > width or h > width):
        r = w/h
        im = im.resize((width, int(width / r)))
    col1.image(im)

    # interpolate
    w, h = im.size
    scale = 4
    col2.write("Bicubic :sparkles:")
    col2.image(im.resize((w*scale, h*scale)))
    im = np.array(im)[:, :, :3]
    if im.max() > 1:
        im = im / 255.
    im = torch.FloatTensor(im[None, :, :, :]).to(st.session_state['model'].device)
    im = im.permute(0, 3, 1, 2)

    col3.write("SR (our model) :sparkles:")
    place = col3.empty()

    place.markdown('![](https://raw.githubusercontent.com/Codelessly/FlutterLoadingGIFs/master/packages/cupertino_activity_indicator_square_small.gif)')

    #with st.spinner('generating...'):
    with torch.no_grad():
        z = st.session_state['model'].netG(im)
        z = z[0].permute(1, 2, 0).detach().numpy()
    place.empty()

    col3.image(z, clamp=True)


    col5.write("SR (nESRGAN+\:CVPR2020) :sparkles:")
    place = col5.empty()

    place.markdown('![](https://raw.githubusercontent.com/Codelessly/FlutterLoadingGIFs/master/packages/cupertino_activity_indicator_square_small.gif)')

    #with st.spinner('generating...'):
    with torch.no_grad():
        z = st.session_state['nESRGANplus'].netG(im)
        #z = st.session_state['100000_G_nESRGANplus'].netG(im)
        z = z[0].permute(1, 2, 0).detach().numpy()
    place.empty()

    col5.image(z, clamp=True)

    #
    col6.write("SR (LIIF\:CVPR2021) :sparkles:")
    place = col6.empty()

    place.markdown('![](https://raw.githubusercontent.com/Codelessly/FlutterLoadingGIFs/master/packages/cupertino_activity_indicator_square_small.gif)')
    with torch.no_grad():
        h, w = z.shape[:2]
        coord = make_coord((h, w))
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        coord = coord.unsqueeze(0)
        cell = cell.unsqueeze(0)
        bsize = 30000

        inp = ((im - 0.5) / 0.5)

        st.session_state['LIIF'].gen_feat(inp)

        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = st.session_state['LIIF'].query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    z = (pred[0] * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).numpy()

    place.empty()
    col6.image(z)



st.set_page_config(layout="wide", page_title="Super Resolution")
col1, col2, col3 = st.columns([1, 3, 3])
col4, col5, col6 = st.columns([1, 3, 3])
my_upload = st.sidebar.file_uploader("Upload an small image please (< 300px)", type=["png", "jpg", "jpeg"])
if my_upload is not None:
    sr(upload=my_upload)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
