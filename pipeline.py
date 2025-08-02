#!/usr/bin/env python
"""
Pipeline Orchestrator für Amazon Reviews Sentimentanalyse.

Dieses Modul implementiert eine vollständige scikit-learn Pipeline, die
alle Preprocessing-Schritte und Vektorisierung zusammenführt.

Verwendung:
    from pipeline import SentimentPipeline
    
    # Pipeline erstellen
    pipeline = SentimentPipeline(vectorizer_type='tfidf')
    
    # Training
    pipeline.fit(train_texts, train_labels)
    
    # Vorhersage
    predictions = pipeline.predict(test_texts)
"""

import pickle
from pathlib import Path
from typing import List, Optional, Union, Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from vectorizer import TFIDFVectorizer, WordPieceVectorizer


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Text Preprocessor für die Pipeline.
    
    Bereinigt und normalisiert Texte für die Vektorisierung.
    """
    
    def __init__(self, lowercase: bool = True, remove_html: bool = True):
        """
        Initialisiert den Text Preprocessor.
        
        Args:
            lowercase: Ob Texte in Kleinbuchstaben umgewandelt werden sollen
            remove_html: Ob HTML-Tags entfernt werden sollen
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        
    def fit(self, X, y=None):
        """Fit-Methode (keine Transformation nötig)."""
        return self
    
    def transform(self, X):
        """
        Transformiert Texte.
        
        Args:
            X: Liste von Texten
            
        Returns:
            Bereinigte Texte
        """
        if isinstance(X, pd.Series):
            X = X.tolist()
        
        processed_texts = []
        for text in X:
            text = str(text)
            
            if self.lowercase:
                text = text.lower()
            
            if self.remove_html:
                import re
                text = re.sub(r'<.*?>', ' ', text)
                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
            
            processed_texts.append(text)
        
        return processed_texts


class VectorizerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper für die Vektorizer, um sie in scikit-learn Pipelines zu verwenden.
    """
    
    def __init__(self, vectorizer_type: Literal['tfidf', 'wordpiece'] = 'tfidf', **kwargs):
        """
        Initialisiert den Vectorizer Wrapper.
        
        Args:
            vectorizer_type: Art des Vektorizers ('tfidf' oder 'wordpiece')
            **kwargs: Weitere Parameter für den Vektorizer
        """
        self.vectorizer_type = vectorizer_type
        self.vectorizer_kwargs = kwargs
        self.vectorizer = None
        
    def fit(self, X, y=None):
        """
        Trainiert den Vektorizer.
        
        Args:
            X: Liste von Texten
            y: Labels (optional)
            
        Returns:
            self
        """
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TFIDFVectorizer(**self.vectorizer_kwargs)
        elif self.vectorizer_type == 'wordpiece':
            self.vectorizer = WordPieceVectorizer(**self.vectorizer_kwargs)
        else:
            raise ValueError(f"Unbekannter Vektorizer-Typ: {self.vectorizer_type}")
        
        self.vectorizer.fit(X)
        return self
    
    def transform(self, X):
        """
        Transformiert Texte in Vektoren.
        
        Args:
            X: Liste von Texten
            
        Returns:
            Vektorisierte Texte
        """
        if self.vectorizer is None:
            raise ValueError("Vektorizer muss zuerst mit fit() trainiert werden")
        
        return self.vectorizer.transform(X)
    
    def get_feature_names_out(self):
        """Gibt Feature-Namen zurück (für scikit-learn Kompatibilität)."""
        if self.vectorizer_type == 'tfidf':
            return np.array(self.vectorizer.get_feature_names())
        else:
            # Für WordPiece geben wir generische Namen zurück
            return np.array([f'wp_feature_{i}' for i in range(self.vectorizer.get_feature_dim())])


class SentimentPipeline:
    """
    Vollständige Pipeline für Sentimentanalyse.
    
    Kombiniert Text-Preprocessing, Vektorisierung und Modell in einer Pipeline.
    """
    
    def __init__(
        self,
        vectorizer_type: Literal['tfidf', 'wordpiece'] = 'tfidf',
        vectorizer_kwargs: Optional[dict] = None,
        model=None,
        random_state: int = 42
    ):
        """
        Initialisiert die Sentiment Pipeline.
        
        Args:
            vectorizer_type: Art des Vektorizers
            vectorizer_kwargs: Parameter für den Vektorizer
            model: scikit-learn Modell (optional)
            random_state: Random Seed
        """
        self.vectorizer_type = vectorizer_type
        self.vectorizer_kwargs = vectorizer_kwargs or {}
        self.model = model
        self.random_state = random_state
        self.pipeline = None
        self.is_fitted = False
        
    def _create_pipeline(self):
        """Erstellt die scikit-learn Pipeline."""
        steps = [
            ('preprocessor', TextPreprocessor()),
            ('vectorizer', VectorizerWrapper(self.vectorizer_type, **self.vectorizer_kwargs))
        ]
        
        if self.model is not None:
            steps.append(('classifier', self.model))
        
        self.pipeline = Pipeline(steps)
    
    def fit(self, X: List[str], y: np.ndarray) -> 'SentimentPipeline':
        """
        Trainiert die Pipeline.
        
        Args:
            X: Liste von Texten
            y: Labels
            
        Returns:
            self
        """
        if self.pipeline is None:
            self._create_pipeline()
        
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Macht Vorhersagen.
        
        Args:
            X: Liste von Texten
            
        Returns:
            Vorhersagen
        """
        if not self.is_fitted:
            raise ValueError("Pipeline muss zuerst mit fit() trainiert werden")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Macht Wahrscheinlichkeitsvorhersagen.
        
        Args:
            X: Liste von Texten
            
        Returns:
            Wahrscheinlichkeiten
        """
        if not self.is_fitted:
            raise ValueError("Pipeline muss zuerst mit fit() trainiert werden")
        
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            raise AttributeError("Modell unterstützt keine Wahrscheinlichkeitsvorhersagen")
    
    def score(self, X: List[str], y: np.ndarray) -> float:
        """
        Berechnet die Accuracy.
        
        Args:
            X: Liste von Texten
            y: Wahre Labels
            
        Returns:
            Accuracy Score
        """
        if not self.is_fitted:
            raise ValueError("Pipeline muss zuerst mit fit() trainiert werden")
        
        return self.pipeline.score(X, y)
    
    def cross_validate(
        self, 
        X: List[str], 
        y: np.ndarray, 
        cv: int = 5
    ) -> dict:
        """
        Führt Cross-Validation durch.
        
        Args:
            X: Liste von Texten
            y: Labels
            cv: Anzahl Folds
            
        Returns:
            Dictionary mit CV-Ergebnissen
        """
        if self.pipeline is None:
            self._create_pipeline()
        
        from sklearn.model_selection import cross_validate
        
        cv_results = cross_validate(
            self.pipeline, X, y, cv=cv, 
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=True
        )
        
        return cv_results
    
    def get_feature_importance(self) -> dict:
        """
        Gibt Feature-Importance zurück (falls verfügbar).
        
        Returns:
            Dictionary mit Feature-Importance
        """
        if not self.is_fitted:
            raise ValueError("Pipeline muss zuerst mit fit() trainiert werden")
        
        if hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            # Für Tree-basierte Modelle
            importances = self.pipeline.named_steps['classifier'].feature_importances_
            feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
            
            return dict(zip(feature_names, importances))
        
        elif hasattr(self.pipeline.named_steps['classifier'], 'coef_'):
            # Für lineare Modelle
            coefficients = self.pipeline.named_steps['classifier'].coef_
            feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
            
            return dict(zip(feature_names, coefficients[0]))
        
        else:
            raise AttributeError("Modell unterstützt keine Feature-Importance")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Speichert die Pipeline.
        
        Args:
            filepath: Pfad zur Speicherdatei
        """
        if not self.is_fitted:
            raise ValueError("Nur trainierte Pipelines können gespeichert werden")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SentimentPipeline':
        """
        Lädt eine gespeicherte Pipeline.
        
        Args:
            filepath: Pfad zur Speicherdatei
            
        Returns:
            Geladene Pipeline
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_baseline_pipeline() -> SentimentPipeline:
    """
    Erstellt eine Baseline-Pipeline mit TF-IDF und Naive Bayes.
    
    Returns:
        SentimentPipeline
    """
    from sklearn.naive_bayes import MultinomialNB
    
    return SentimentPipeline(
        vectorizer_type='tfidf',
        vectorizer_kwargs={
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95
        },
        model=MultinomialNB()
    )


def create_advanced_pipeline() -> SentimentPipeline:
    """
    Erstellt eine fortgeschrittene Pipeline mit WordPiece und SVM.
    
    Returns:
        SentimentPipeline
    """
    from sklearn.svm import LinearSVC
    
    return SentimentPipeline(
        vectorizer_type='wordpiece',
        vectorizer_kwargs={
            'max_length': 128,
            'aggregation': 'mean'
        },
        model=LinearSVC(random_state=42, max_iter=1000)
    )


def main():
    """
    Beispiel für die Verwendung der Pipeline.
    """
    print("🔧 Pipeline Orchestrator für Sentimentanalyse")
    print("=" * 60)
    
    # Daten laden
    from vectorizer import load_processed_data, extract_texts_and_labels
    
    data_dir = Path("data/processed")
    splits = load_processed_data(data_dir)
    
    if 'train' not in splits:
        print("❌ Trainingsdaten nicht gefunden!")
        return
    
    # Kleine Stichprobe für Demo
    sample_df = splits['train'].sample(n=1000, random_state=42)
    train_texts, train_labels = extract_texts_and_labels(sample_df)
    
    print(f"📊 Trainingsdaten: {len(train_texts)} Samples")
    print(f"📊 Label-Verteilung: {np.bincount(train_labels)}")
    
    # Baseline Pipeline testen
    print(f"\n🔤 Baseline Pipeline (TF-IDF + Naive Bayes):")
    baseline_pipeline = create_baseline_pipeline()
    
    # Cross-Validation
    cv_results = baseline_pipeline.cross_validate(train_texts, train_labels, cv=3)
    print(f"✅ CV Accuracy: {cv_results['test_accuracy'].mean():.3f} (+/- {cv_results['test_accuracy'].std() * 2:.3f})")
    print(f"✅ CV F1-Score: {cv_results['test_f1_macro'].mean():.3f}")
    
    # Pipeline trainieren
    baseline_pipeline.fit(train_texts, train_labels)
    
    # Speichern
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    baseline_pipeline.save(models_dir / "baseline_pipeline.pkl")
    
    print(f"✅ Baseline Pipeline gespeichert: {models_dir / 'baseline_pipeline.pkl'}")
    
    # Advanced Pipeline (nur mit kleineren Daten für Demo)
    print(f"\n🔤 Advanced Pipeline (WordPiece + SVM):")
    sample_size = min(500, len(train_texts))
    advanced_pipeline = create_advanced_pipeline()
    advanced_pipeline.fit(train_texts[:sample_size], train_labels[:sample_size])
    
    # Speichern
    advanced_pipeline.save(models_dir / "advanced_pipeline.pkl")
    print(f"✅ Advanced Pipeline gespeichert: {models_dir / 'advanced_pipeline.pkl'}")
    
    print(f"\n🎉 Pipeline Orchestrator ist bereit für das Training!")


if __name__ == "__main__":
    main()