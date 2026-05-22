"""
model_loader.py
===============
Makine öğrenmesi ve derin öğrenme modellerini yükleyen modül.
Streamlit cache mekanizması ile modeller yalnızca bir kez yüklenir.
"""

import pickle
import streamlit as st
from yaml import warnings


class ModelLoader:
    """
    ML ve DL modellerini yükleyen sınıf.

    Streamlit'in @st.cache_resource dekoratörü sayesinde modeller
    yalnızca bir kez yüklenir; sonraki çağrılarda önbellekten döner.
    """

    @staticmethod
    @st.cache_resource
    def load_classic_models() -> tuple:
        """
        Klasik ML modellerini ve TF-IDF vektörleyiciyi yükler.

        Döndürür
        --------
        tuple : (RandomForestClassifier, LogisticRegression, TfidfVectorizer)

        Raises
        ------
        FileNotFoundError
            Model dosyası bulunamazsa.
        """
        with open('models/rf_model.pkl', 'rb') as f:
            rf = pickle.load(f)
        with open('models/lr_model.pkl', 'rb') as f:
            lr = pickle.load(f)
        with open('models/tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        return rf, lr, tfidf

    @staticmethod
    @st.cache_resource
    def load_berturk() -> tuple:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        """
        BERTurk modelini ve tokenizer'ı yükler.

        Döndürür
        --------
        tuple : (AutoTokenizer, AutoModelForSequenceClassification)

        Raises
        ------
        OSError
            Model dosyaları bulunamazsa.
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained('enesmidesem/spamkiller-berturk')
        model = AutoModelForSequenceClassification.from_pretrained('enesmidesem/spamkiller-berturk')
        model.eval()
        return tokenizer, model