"""
ui.py
=====
Streamlit arayüzünü oluşturan ve yöneten modül.
Koyu tema, gradient kartlar ve modern tasarım ile güçlendirilmiştir.
"""

import streamlit as st

from src.preprocessor import TextPreprocessor
from src.model_loader import ModelLoader
from src.predictor import SpamPredictor


# ── Özel CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Genel arka plan ── */
.stApp {
    background: #0d0f14;
    font-family: 'DM Sans', sans-serif;
}

/* ── Üst toolbar gizle ── */
header[data-testid="stHeader"] { background: transparent; }
.stDeployButton { display: none; }

/* ── Ana başlık ── */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e8f4f8 0%, #7eb8d4 50%, #3a7bd5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #6b7a8d;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, #3a7bd5 0%, #7eb8d4 40%, transparent 100%);
    margin: 1.5rem 0 2rem 0;
    border: none;
}

/* ── Model bilgi kartı ── */
.model-card {
    background: linear-gradient(135deg, #141820 0%, #1a2030 100%);
    border: 1px solid #2a3448;
    border-left: 3px solid #3a7bd5;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0 1.5rem 0;
    color: #c8d8e8;
    font-size: 0.9rem;
}
.model-card strong { color: #7eb8d4; }

/* ── Input alanları ── */
.stTextArea textarea {
    background: #141820 !important;
    border: 1px solid #2a3448 !important;
    border-radius: 10px !important;
    color: #c8d8e8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: #3a7bd5 !important;
    box-shadow: 0 0 0 2px rgba(58, 123, 213, 0.15) !important;
}

/* ── File uploader ── */
.stFileUploader {
    background: #141820;
    border: 1px dashed #2a3448;
    border-radius: 10px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
.stFileUploader:hover { border-color: #3a7bd5; }
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
}

/* ── Analiz Et butonu ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e40af 0%, #3a7bd5 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(58, 123, 213, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(58, 123, 213, 0.5) !important;
}

/* ── Sonuç kartları ── */
.result-spam {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d1010 100%);
    border: 1px solid #5a1a1a;
    border-left: 4px solid #e53e3e;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin: 1rem 0;
}
.result-real {
    background: linear-gradient(135deg, #0a1a0f 0%, #102d18 100%);
    border: 1px solid #1a5a2a;
    border-left: 4px solid #38a169;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin: 1rem 0;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
}
.result-label.spam { color: #fc8181; }
.result-label.real { color: #68d391; }
.result-sub {
    color: #6b7a8d;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Güven skoru bar ── */
.confidence-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.2rem 0 0.5rem 0;
}
.confidence-label {
    color: #7eb8d4;
    font-size: 0.85rem;
    font-family: 'Space Mono', monospace;
    min-width: 90px;
}
.confidence-bar-bg {
    flex: 1;
    height: 6px;
    background: #1e2a3a;
    border-radius: 99px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #3a7bd5, #7eb8d4);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.confidence-pct {
    color: #c8d8e8;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    min-width: 48px;
    text-align: right;
}

/* ── Metrik kartları ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.metric-card {
    background: #141820;
    border: 1px solid #2a3448;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-name {
    color: #4a6080;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.35rem;
}
.metric-value {
    color: #7eb8d4;
    font-size: 1.35rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}
.metric-unit {
    color: #3a7bd5;
    font-size: 0.7rem;
}

/* ── Olasılık satırları ── */
.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1a2030;
    color: #8899aa;
    font-size: 0.88rem;
}
.prob-row:last-child { border-bottom: none; }
.prob-val { color: #c8d8e8; font-family: 'Space Mono', monospace; }

/* ── Şüpheli kelimeler ── */
.suspicious-section {
    background: #141820;
    border: 1px solid #2a3448;
    border-radius: 10px;
    padding: 1.25rem;
    margin-top: 1rem;
}
.suspicious-title {
    color: #7eb8d4;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.email-text {
    color: #8899aa;
    font-size: 0.92rem;
    line-height: 1.8;
    word-break: break-word;
}

/* ── Bölüm başlığı ── */
.section-label {
    color: #4a6080;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin: 1.5rem 0 0.75rem 0;
}

/* ── Radio butonlar ── */
.stRadio > div { gap: 0.5rem; }
.stRadio label {
    background: #141820 !important;
    border: 1px solid #2a3448 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    color: #7eb8d4 !important;
    transition: all 0.15s ease !important;
    font-size: 0.88rem !important;
}
.stRadio label:hover {
    border-color: #3a7bd5 !important;
    background: #1a2235 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3a7bd5 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a3448; border-radius: 3px; }
</style>
"""


class SpamKillerUI:
    """
    Streamlit arayüzünü oluşturan ve yöneten sınıf.

    Parametreler
    ------------
    predictor    : SpamPredictor
    loader       : ModelLoader
    preprocessor : TextPreprocessor
    """

    MODEL_INFO: dict = {
        "Random Forest":      ("🌲", "En az yanlış alarm — %99.25 Precision"),
        "Lojistik Regresyon": ("📈", "Hızlı ve yorumlanabilir — %96.77 Accuracy"),
        "BERTurk":            ("🧠", "En az spam kaçırır — %95.97 Recall"),
    }

    MODEL_METRICS: dict = {
        "Random Forest": {
            "Accuracy": 98.47, "Precision": 99.25,
            "Recall": 89.26,   "F1": 93.99,
        },
        "Lojistik Regresyon": {
            "Accuracy": 96.77, "Precision": 92.48,
            "Recall": 82.55,   "F1": 87.23,
        },
        "BERTurk": {
            "Accuracy": 98.56, "Precision": 93.46,
            "Recall": 95.97,   "F1": 94.70,
        },
    }

    def __init__(
        self,
        predictor: SpamPredictor,
        loader: ModelLoader,
        preprocessor: TextPreprocessor,
    ) -> None:
        self.predictor    = predictor
        self.loader       = loader
        self.preprocessor = preprocessor

    # ── Yardımcı render metodları ────────────────────────────────────────────

    def _inject_css(self) -> None:
        """Özel CSS'i sayfaya enjekte eder."""
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    def render_header(self) -> None:
        """Hero başlık bölümünü render eder."""
        st.markdown(
            '<h1 class="hero-title">⚡ SpamKiller</h1>'
            '<p class="hero-sub">Yapay Zeka Destekli Türkçe Spam Tespit Sistemi</p>'
            '<hr class="hero-divider">',
            unsafe_allow_html=True,
        )

    def render_model_selector(self) -> str:
        """
        Model seçim alanını render eder.

        Döndürür
        --------
        str : Seçilen model adı.
        """
        st.markdown('<p class="section-label">Model Seç</p>', unsafe_allow_html=True)
        model_choice = st.radio(
            label="Model Seçimi",
            options=list(self.MODEL_INFO.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        icon, desc = self.MODEL_INFO[model_choice]
        st.markdown(
            f'<div class="model-card">{icon} &nbsp;<strong>{model_choice}</strong> — {desc}</div>',
            unsafe_allow_html=True,
        )
        return model_choice

    def render_input(self) -> str:
        """
        Dosya yükleme ve metin giriş alanını render eder.
        Dosya yüklenirse text area'nın önüne geçer.

        Döndürür
        --------
        str : Analiz edilecek metin.

        Zaman Karmaşıklığı: O(n) — n: dosya/metin boyutu
        """
        st.markdown('<p class="section-label">E-posta Metni</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "📂 .txt dosyası yükle",
            type=["txt"],
            label_visibility="collapsed",
        )
        if uploaded:
            text = uploaded.read().decode("utf-8", errors="ignore")
            st.text_area("Yüklenen içerik", value=text, height=160, disabled=True, label_visibility="collapsed")
            return text

        return st.text_area(
            label="E-posta Metni",
            height=180,
            placeholder="Buraya e-posta metnini yapıştırın…\n\nÖrnek: Tebrikler! 50.000 TL ikramiye kazandınız.Paranızı hesabınıza çekmek için hemen tıklayın: http://spamlink.com",
            label_visibility="collapsed",
        )

    def render_metrics(self, model_choice: str) -> None:
        """
        Seçilen modelin eğitim metriklerini kart grid olarak render eder.

        Parametreler
        ------------
        model_choice : str
        """
        m = self.MODEL_METRICS[model_choice]
        st.markdown('<p class="section-label">Model Performans Metrikleri</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-name">Accuracy</div>
                    <div class="metric-value">{m['Accuracy']:.1f}<span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Precision</div>
                    <div class="metric-value">{m['Precision']:.1f}<span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Recall</div>
                    <div class="metric-value">{m['Recall']:.1f}<span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">F1-Score</div>
                    <div class="metric-value">{m['F1']:.1f}<span class="metric-unit">%</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_result(
        self,
        prediction: int,
        confidence: float,
        probabilities,
        suspicious: list,
        original_text: str,
        model_choice: str,
    ) -> None:
        """
        Tahmin sonucunu, metrikleri ve şüpheli kelimeleri render eder.

        Parametreler
        ------------
        prediction    : int
        confidence    : float
        probabilities : ndarray
        suspicious    : list
        original_text : str
        model_choice  : str
        """
        # Sonuç kartı
        if prediction == 1:
            st.markdown(
                '<div class="result-spam">'
                '<p class="result-label spam">🚨 SPAM E-POSTA</p>'
                '<p class="result-sub">Bu mesaj spam olarak sınıflandırıldı</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-real">'
                '<p class="result-label real">✅ GERÇEK E-POSTA</p>'
                '<p class="result-sub">Bu mesaj güvenli olarak sınıflandırıldı</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Güven skoru
        bar_color = "#e53e3e" if prediction == 1 else "#38a169"
        st.markdown(
            f"""
            <div class="confidence-row">
                <span class="confidence-label">Güven</span>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence:.1f}%;background:linear-gradient(90deg,{bar_color},{bar_color}99);"></div>
                </div>
                <span class="confidence-pct">%{confidence:.1f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Olasılık detayı
        with st.expander("📊 Detaylı Olasılıklar"):
            st.markdown(
                f"""
                <div class="prob-row">
                    <span>Gerçek E-posta</span>
                    <span class="prob-val">%{float(probabilities[0])*100:.1f}</span>
                </div>
                <div class="prob-row">
                    <span>Spam</span>
                    <span class="prob-val">%{float(probabilities[1])*100:.1f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Metrikler
        self.render_metrics(model_choice)

        # Şüpheli kelimeler
        if prediction == 1:
            if suspicious:
                highlighted = self.preprocessor.highlight(original_text, suspicious)
                st.markdown(
                    f'<div class="suspicious-section">'
                    f'<p class="suspicious-title">🔎 Şüpheli Kelimeler</p>'
                    f'<div class="email-text">{highlighted}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif model_choice == "BERTurk":
                st.markdown(
                    '<div class="suspicious-section">'
                    '<p class="suspicious-title">🧠 Not</p>'
                    '<p class="email-text">BERTurk bağlam tabanlı çalıştığı için kelime bazlı vurgulama yapılamamaktadır.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    def run(self) -> None:
        """
        Uygulamayı başlatır ve ana döngüyü yönetir.
        Streamlit her etkileşimde bu metodu yeniden çalıştırır.
        """
        self._inject_css()
        self.render_header()
        model_choice = self.render_model_selector()
        email_text   = self.render_input()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍  ANALİZ ET", type="primary", use_container_width=True):
            if not email_text or not email_text.strip():
                st.warning("Lütfen bir e-posta metni girin veya dosya yükleyin.")
                return

            with st.spinner("Analiz ediliyor..."):
                if model_choice == "BERTurk":
                    tokenizer, bert_model = self.loader.load_berturk()
                    prediction, confidence, probabilities, suspicious = \
                        self.predictor.predict_berturk(email_text, tokenizer, bert_model)
                elif model_choice == "Random Forest":
                    rf, _, tfidf = self.loader.load_classic_models()
                    prediction, confidence, probabilities, suspicious = \
                        self.predictor.predict_classic(email_text, rf, tfidf)
                else:
                    _, lr, tfidf = self.loader.load_classic_models()
                    prediction, confidence, probabilities, suspicious = \
                        self.predictor.predict_classic(email_text, lr, tfidf)

            st.markdown('<p class="section-label">Analiz Sonucu</p>', unsafe_allow_html=True)
            self.render_result(
                prediction, confidence, probabilities,
                suspicious, email_text, model_choice,
            )