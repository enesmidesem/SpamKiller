"""
preprocessor.py
===============
Türkçe e-posta metinlerini makine öğrenmesi için hazırlayan ön işleme modülü.
"""

import re


class TextPreprocessor:
    """
    Türkçe e-posta metinlerini temizleyen ve normalize eden sınıf.

    Özellikler
    ----------
    - URL, e-posta adresi ve sayıları özel token'larla değiştirir.
    - Noktalama işaretlerini temizler.
    - Türkçe stopword'leri O(1) karmaşıklıkla filtreler (frozenset).
    - Regex pattern'ları sınıf seviyesinde bir kez derlenir.

    Zaman Karmaşıklığı
    ------------------
    clean()     : O(n)       — n: metin uzunluğu
    highlight() : O(k * n)   — k: şüpheli kelime sayısı (genellikle ≤ 5)
    """

    # Stopword listesi frozenset → O(1) arama karmaşıklığı
    STOPWORDS: frozenset = frozenset({
        'bir', 've', 'ile', 'bu', 'da', 'de', 'mi', 'mu', 'mı', 'mü',
        'için', 'ama', 'fakat', 'ya', 'ki', 'ne', 'ben', 'sen', 'o',
        'biz', 'siz', 'onlar', 'gibi', 'kadar', 'daha', 'çok', 'en',
        'var', 'yok', 'olan', 'olarak', 'diye', 'bile', 'her',
        'hiç', 'hiçbir', 'bazı', 'tüm', 'bütün', 'ise', 'veya', 'hem'
    })

    # Regex pattern'ları bir kez derlenir → tekrar kullanımda O(1)
    _URL_PATTERN    = re.compile(r'http\S+|www\S+')
    _EMAIL_PATTERN  = re.compile(r'\S+@\S+')
    _NUMBER_PATTERN = re.compile(r'\d+')
    _PUNCT_PATTERN  = re.compile(r'[^\w\s]')
    _SPACES_PATTERN = re.compile(r'\s+')

    def clean(self, text: str) -> str:
        """
        Metni temizler ve normalize eder.

        Parametreler
        ------------
        text : str
            Ham e-posta metni.

        Döndürür
        --------
        str
            Temizlenmiş, normalize edilmiş metin.

        Zaman Karmaşıklığı: O(n)
        """
        text = str(text).lower()
        text = self._URL_PATTERN.sub('[URL]', text)
        text = self._EMAIL_PATTERN.sub('[EMAIL]', text)
        text = self._NUMBER_PATTERN.sub('[SAYI]', text)
        text = self._PUNCT_PATTERN.sub(' ', text)
        text = self._SPACES_PATTERN.sub(' ', text).strip()
        words = [w for w in text.split() if w not in self.STOPWORDS]
        return ' '.join(words)

    def highlight(self, text: str, suspicious_words: list) -> str:
        """
        Şüpheli kelimeleri HTML ile renkli olarak işaretler.

        Parametreler
        ------------
        text : str
            Orijinal e-posta metni.
        suspicious_words : list[str]
            Vurgulanacak kelimeler (genellikle ≤ 5 adet).

        Döndürür
        --------
        str
            HTML işaretleri içeren metin.

        Zaman Karmaşıklığı: O(k * n)
        """
        style = (
            'background-color:#ff4b4b;color:white;'
            'padding:2px 6px;border-radius:4px;font-weight:bold;'
        )
        for word in suspicious_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub(f'<mark style="{style}">{word}</mark>', text)
        return text