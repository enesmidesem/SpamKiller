"""
predictor.py
============
Spam tahmin işlemlerini yürüten modül.
Klasik ML modelleri ve BERTurk için ayrı tahmin metodları içerir.
"""

from src.preprocessor import TextPreprocessor


class SpamPredictor:
    """
    Spam tahmin işlemlerini yürüten sınıf.

    Parametreler
    ------------
    preprocessor : TextPreprocessor
        Metin ön işleme nesnesi.
    """

    def __init__(self, preprocessor: TextPreprocessor) -> None:
        """
        SpamPredictor nesnesini başlatır.

        Parametreler
        ------------
        preprocessor : TextPreprocessor
            Metin ön işleme nesnesi.
        """
        self.preprocessor = preprocessor

    def predict_classic(self, text: str, model, tfidf) -> tuple:
        """
        Klasik ML modeli ile spam tahmini yapar.

        Parametreler
        ------------
        text : str
            Ham e-posta metni.
        model : sklearn estimator
            RandomForestClassifier veya LogisticRegression.
        tfidf : TfidfVectorizer
            Eğitilmiş TF-IDF vektörleyici.

        Döndürür
        --------
        tuple : (tahmin: int, güven: float, olasılık: ndarray, şüpheli: list)
            - tahmin   : 0 (gerçek) veya 1 (spam)
            - güven    : yüzde cinsinden güven skoru
            - olasılık : [gerçek_olasılık, spam_olasılık]
            - şüpheli  : en etkili 5 kelime

        Zaman Karmaşıklığı: O(n) — TF-IDF dönüşümü + model tahmini
        """
        cleaned = self.preprocessor.clean(text)
        vector = tfidf.transform([cleaned])

        prediction    = int(model.predict(vector)[0])
        probabilities = model.predict_proba(vector)[0]
        confidence    = float(probabilities[prediction] * 100)

        # Şüpheli kelimeleri bul — sadece aktif (sıfır olmayan) özellikler: O(nnz)
        feature_names  = tfidf.get_feature_names_out()
        active_indices = vector.nonzero()[1]

        if hasattr(model, 'feature_importances_'):   # Random Forest
            importances = model.feature_importances_
        else:                                         # Lojistik Regresyon
            importances = abs(model.coef_[0])

        scores    = {feature_names[i]: importances[i] for i in active_indices}
        suspicious = sorted(scores, key=scores.get, reverse=True)[:5]

        return prediction, confidence, probabilities, suspicious

    def predict_berturk(self, text: str, tokenizer, model) -> tuple:
        """
        BERTurk modeli ile spam tahmini yapar.

        Parametreler
        ------------
        text : str
            Ham e-posta metni (BERTurk kendi tokenization'ını yapar).
        tokenizer : AutoTokenizer
            BERTurk tokenizer.
        model : AutoModelForSequenceClassification
            Fine-tune edilmiş BERTurk modeli.

        Döndürür
        --------
        tuple : (tahmin: int, güven: float, olasılık: ndarray, şüpheli: list)
            şüpheli her zaman boş listedir (BERTurk kelime bazlı değil).

        Zaman Karmaşıklığı: O(L²) — L: token sayısı (Transformer dikkat mekanizması)
        """
        import torch

        encoding = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        with torch.no_grad():
            outputs = model(**encoding)

        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        prediction    = int(torch.argmax(probabilities).item())
        confidence    = float(probabilities[prediction].item() * 100)

        return prediction, confidence, probabilities.numpy(), []