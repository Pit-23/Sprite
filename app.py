import os
import requests

'''
def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {filepath} ...")
        r = requests.get(url, allow_redirects=True)
        with open(filepath, 'wb') as f:
            f.write(r.content)
        print(f"Downloaded {filepath}")

os.makedirs("models", exist_ok=True)

# GFPGAN
download_file(
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "models/GFPGANv1.4.pth"
)

# InSwapper
download_file(
    "https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx",
    "models/inswapper_128.onnx"
)
'''

import streamlit as st
import numpy as np
import cv2
#import insightface
#from insightface.app import FaceAnalysis
#from insightface.model_zoo import get_model
#import torch
from PIL import Image, ImageDraw, ImageFont

#import torchvision.transforms.functional as F
import sys
import types

# Patch: emulate the old module path expected by basicsr
#sys.modules['torchvision.transforms.functional_tensor'] = types.ModuleType('torchvision.transforms.functional_tensor')
#sys.modules['torchvision.transforms.functional_tensor'].rgb_to_grayscale = F.rgb_to_grayscale

#from gfpgan import GFPGANer

# Load models
#swapper = get_model('models/inswapper_128.onnx', download=False, providers=['CPUExecutionProvider'])
#app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
#app.prepare(ctx_id=0, det_size=(640,640))
#restorer = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2)

st.title("Sprite Personalized Poster Generator")
st.warning("This demo only accepts user uploaded images of yourself. Do not upload celebrity/public figure photos.")

name = st.text_input("Nama")
place = st.text_input("Nama Warung")
place = place.upper()
address = st.text_input("Alamat")
options = ["Jakarta","Surabaya"]
city = st.selectbox("Kota", options)
if city == "Jakarta" : 
  option_daerah = ["Jakarta Utara", "Jakarta Selatan"]
  daerah = st.selectbox("Kecamatan", option_daerah)
  if daerah  == "Jakarta Utara" : 
    option_kecamatan = ["PENJARINGAN", "PADEMANGAN", "TANJUNG PRIOK", "KOJA", "CILINCING", "KELAPA GADING"]
    kecamatan = st.selectbox("Kecamatan", option_kecamatan)
  else : 
    option_kecamatan = ["Cilandak", "Jagakarsa", "Kebayoran Baru", "Kebayoran Lama", 
                        "Mampang Prapatan", "Pancoran", "Pasar Minggu", "Pesanggrahan", "Setiabudi", "Tebet"]
    kecamatan = st.selectbox("Kecamatan", option_kecamatan)

elif city == "Surabaya" :
  option_daerah = ["Surabaya Pusat", "Surabaya Selatan"]
  daerah = st.selectbox("Daerah", option_daerah)
  if daerah  == "Surabaya Pusat" : 
    option_kecamatan = ["Genteng", "Tegalsari", "Bubutan", "Simokerto"]
    kecamatan = st.selectbox("Kecamatan", option_kecamatan)
  else :
    option_kecamatan = ["Sawahan", "Wonokromo", "Dukuh Pakis", "Karangpilang", 
                        "Wiyung", "Wonocolo", "Gayungan", "Jambangan"]
    kecamatan = st.selectbox("Kecamatan", option_kecamatan)

src_file = st.file_uploader("Upload Foto Selfie", type=['jpg', 'jpeg', 'png'])

if src_file:
    src_bytes = np.frombuffer(src_file.read(), np.uint8)

    src_img = cv2.imdecode(src_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB),
          caption=["Source"], width=300)
    faces = app.get(src_img)
    gen = faces[0].gender

    if gen == 1:
        st.text("Male face detected")
        tgt_img = cv2.imread('Foto_Cowok (2).jpg', cv2.IMREAD_COLOR)
    else:
        st.text("Female face detected")
        tgt_img = cv2.imread('Foto_Cewek (1).jpg', cv2.IMREAD_COLOR)


    if st.button("Go"):
      faces_target = app.get(tgt_img)
      faces_source = app.get(src_img)

      source_face = faces_source[0]
      target_face = faces_target[0]

      swapped = swapper.get(tgt_img, target_face, source_face, paste_back=True)
      _, _, restored_img = restorer.enhance(swapped, has_aligned=False, only_center_face=False, paste_back=True)

      
      base = cv2.imread("Sprite.png")

      overlay = cv2.resize(restored_img, (1000, 400))

      y, x = 115,50

      base[y:y+overlay.shape[0], x:x+overlay.shape[1]] = overlay

      img_pil = Image.fromarray(base)

      draw = ImageDraw.Draw(img_pil)

      font = ImageFont.truetype("AktivGrotesk-Bold.ttf", 76)   # adjust 90 px font size

      draw.text((1125, 225), place, font=font, fill="#39B75A")

      # convert back to cv2 if you want to continue process
      base = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

      Image.MAX_IMAGE_PIXELS = 500000000

      st.image(cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR), caption="Final Enhanced Result", use_container_width=True)
      st.image(base, caption="Final Result", use_container_width=True)
