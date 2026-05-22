"""
app.py
======
SpamKiller uygulamasının giriş noktası.
Tüm bileşenleri bir araya getirir ve uygulamayı başlatır.
"""

import streamlit as st

st.set_page_config(
    page_title="SpamKiller",
    page_icon="🛡️",
    layout="centered"
)

from src.preprocessor import TextPreprocessor
from src.model_loader import ModelLoader
from src.predictor import SpamPredictor
from src.ui import SpamKillerUI

preprocessor = TextPreprocessor()
predictor    = SpamPredictor(preprocessor)
loader       = ModelLoader()
app          = SpamKillerUI(predictor, loader, preprocessor)
app.run()