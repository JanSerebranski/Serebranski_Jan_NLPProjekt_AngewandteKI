#!/usr/bin/env python
"""
Feature Engineering Module für Amazon Reviews Sentimentanalyse.

Dieses Modul implementiert TF-IDF und WordPiece Vektorisierung für die
Vorhersage von Sternebewertungen (1-5) aus Rezensionstexten.

Verwendung:
    from vectorizer import TFIDFVectorizer, WordPieceVectorizer
    
    # TF-IDF Vektorisierung
    tfidf = TFIDFVectorizer()
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)
    
    # WordPiece Vektorisierung
    wp = WordPieceVectorizer()
    X_train = wp.fit_transform(train_texts)
    X_test = wp.transform(test_texts)
"""

import json
import pickle
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import torch


class TFIDFVectorizer:
    """
    TF-IDF Vektorisierer für Textklassifikation.
    
    Wrapper um scikit-learn's TfidfVectorizer mit zusätzlichen
    Funktionen für Persistierung und Metadaten.
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: tuple = (1, 2),
        stop_words: Optional[str] = 'english',
        random_state: int = 42
    ):
        """
        Initialisiert den TF-IDF Vektorizer.
        
        Args:
            max_features: Maximale Anzahl Features
            min_df: Minimale Dokumentfrequenz
            max_df: Maximale Dokumentfrequenz
            ngram_range: N-Gramm Bereich (1,2) = Unigramme + Bigramme
            stop_words: Stopwords-Sprache oder None
            random_state: Random Seed für Reproduzierbarkeit (wird ignoriert)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=stop_words
        )
        self.is_fitted = False
        self.feature_names = None
        self.vocabulary_size = None
        
    def fit(self, texts: List[str]) -> 'TFIDFVectorizer':
        """
        Trainiert den Vektorisierer auf den Trainingsdaten.
        
        Args:
            texts: Liste von Texten
            
        Returns:
            self für Method Chaining
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transformiert Texte in TF-IDF Vektoren.
        
        Args:
            texts: Liste von Texten
            
        Returns:
            TF-IDF Matrix (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Vektorizer muss zuerst mit fit() trainiert werden")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Kombiniert fit() und transform() in einem Schritt.
        
        Args:
            texts: Liste von Texten
            
        Returns:
            TF-IDF Matrix (n_samples, n_features)
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Gibt die Feature-Namen zurück."""
        if not self.is_fitted:
            raise ValueError("Vektorizer muss zuerst mit fit() trainiert werden")
        return self.feature_names.tolist()
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """
        Gibt die Top-N Features nach IDF-Gewichtung zurück.
        
        Args:
            n: Anzahl der Top-Features
            
        Returns:
            Liste der Top-N Feature-Namen
        """
        if not self.is_fitted:
            raise ValueError("Vektorizer muss zuerst mit fit() trainiert werden")
        
        # IDF-Werte sortieren
        idf_scores = self.vectorizer.idf_
        top_indices = np.argsort(idf_scores)[-n:]
        return [self.feature_names[i] for i in top_indices]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Speichert den trainierten Vektorizer.
        
        Args:
            filepath: Pfad zur Speicherdatei
        """
        if not self.is_fitted:
            raise ValueError("Nur trainierte Vektorizer können gespeichert werden")
        
        data = {
            'vectorizer': self.vectorizer,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'vocabulary_size': self.vocabulary_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TFIDFVectorizer':
        """
        Lädt einen gespeicherten Vektorizer.
        
        Args:
            filepath: Pfad zur Speicherdatei
            
        Returns:
            Geladener TFIDFVectorizer
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.vectorizer = data['vectorizer']
        instance.is_fitted = data['is_fitted']
        instance.feature_names = data['feature_names']
        instance.vocabulary_size = data['vocabulary_size']
        
        return instance


class WordPieceVectorizer:
    """
    WordPiece Vektorisierer basierend auf BERT Tokenizer.
    
    Verwendet Hugging Face Transformers für moderne Tokenisierung
    und erstellt Dense-Vektoren für klassische ML-Modelle.
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = True,
        return_tensors: str = 'pt',
        aggregation: str = 'mean'
    ):
        """
        Initialisiert den WordPiece Vektorizer.
        
        Args:
            model_name: Hugging Face Modell-Name
            max_length: Maximale Sequenzlänge
            truncation: Ob Texte abgeschnitten werden sollen
            padding: Ob Padding hinzugefügt werden soll
            return_tensors: Tensor-Format ('pt', 'np', 'tf')
            aggregation: Aggregationsmethode ('mean', 'sum', 'cls')
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.return_tensors = return_tensors
        self.aggregation = aggregation
        self.is_fitted = False
        self.feature_dim = None
        
    def _tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenisiert eine Liste von Texten.
        
        Args:
            texts: Liste von Texten
            
        Returns:
            Tokenizer-Output Dictionary
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors=self.return_tensors
        )
    
    def _aggregate_embeddings(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> np.ndarray:
        """
        Aggregiert Token-Embeddings zu Dokument-Vektoren.
        
        Args:
            input_ids: Token-IDs
            attention_mask: Attention Mask
            
        Returns:
            Aggregierte Dokument-Vektoren
        """
        # Einfache Aggregation: Mittelwert über alle Tokens
        if self.aggregation == 'mean':
            # Attention mask anwenden und Mittelwert berechnen
            masked_ids = input_ids * attention_mask
            sum_embeddings = torch.sum(masked_ids, dim=1)
            token_counts = torch.sum(attention_mask, dim=1, keepdim=True)
            mean_embeddings = sum_embeddings / token_counts
            return mean_embeddings.numpy()
        
        elif self.aggregation == 'sum':
            masked_ids = input_ids * attention_mask
            return torch.sum(masked_ids, dim=1).numpy()
        
        elif self.aggregation == 'cls':
            # CLS Token (erste Position) verwenden
            return input_ids[:, 0].numpy()
        
        else:
            raise ValueError(f"Unbekannte Aggregationsmethode: {self.aggregation}")
    
    def fit(self, texts: List[str]) -> 'WordPieceVectorizer':
        """
        Trainiert den Vektorizer (hier nur Metadaten).
        
        Args:
            texts: Liste von Texten
            
        Returns:
            self für Method Chaining
        """
        # Für WordPiece ist kein Training nötig, aber wir können
        # die Feature-Dimension bestimmen
        sample_tokens = self._tokenize_texts(texts[:1])
        self.feature_dim = sample_tokens['input_ids'].shape[1]
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transformiert Texte in WordPiece Vektoren.
        
        Args:
            texts: Liste von Texten
            
        Returns:
            WordPiece Vektoren (n_samples, max_length)
        """
        if not self.is_fitted:
            raise ValueError("Vektorizer muss zuerst mit fit() trainiert werden")
        
        tokens = self._tokenize_texts(texts)
        return self._aggregate_embeddings(
            tokens['input_ids'], 
            tokens['attention_mask']
        )
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Kombiniert fit() und transform() in einem Schritt.
        
        Args:
            texts: Liste von Texten
            
        Returns:
            WordPiece Vektoren (n_samples, max_length)
        """
        return self.fit(texts).transform(texts)
    
    def get_vocabulary_size(self) -> int:
        """Gibt die Größe des Vokabulars zurück."""
        return self.tokenizer.vocab_size
    
    def get_feature_dim(self) -> int:
        """Gibt die Feature-Dimension zurück."""
        if not self.is_fitted:
            raise ValueError("Vektorizer muss zuerst mit fit() trainiert werden")
        return self.feature_dim
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Speichert den trainierten Vektorizer.
        
        Args:
            filepath: Pfad zur Speicherdatei
        """
        if not self.is_fitted:
            raise ValueError("Nur trainierte Vektorizer können gespeichert werden")
        
        # Tokenizer speichern
        tokenizer_path = Path(filepath).parent / f"{Path(filepath).stem}_tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Metadaten speichern
        data = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'truncation': self.truncation,
            'padding': self.padding,
            'return_tensors': self.return_tensors,
            'aggregation': self.aggregation,
            'is_fitted': self.is_fitted,
            'feature_dim': self.feature_dim,
            'tokenizer_path': str(tokenizer_path)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'WordPieceVectorizer':
        """
        Lädt einen gespeicherten Vektorizer.
        
        Args:
            filepath: Pfad zur Speicherdatei
            
        Returns:
            Geladener WordPieceVectorizer
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Tokenizer laden
        tokenizer_path = data['tokenizer_path']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        instance = cls(
            model_name=data['model_name'],
            max_length=data['max_length'],
            truncation=data['truncation'],
            padding=data['padding'],
            return_tensors=data['return_tensors'],
            aggregation=data['aggregation']
        )
        instance.tokenizer = tokenizer
        instance.is_fitted = data['is_fitted']
        instance.feature_dim = data['feature_dim']
        
        return instance


def load_processed_data(data_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Lädt die verarbeiteten Train/Valid/Test Daten.
    
    Args:
        data_dir: Pfad zum Datenverzeichnis
        
    Returns:
        Dictionary mit Train/Valid/Test DataFrames
    """
    data_dir = Path(data_dir)
    splits = {}
    
    # Korrekte Pfade zu den zusammengeführten Dateien
    split_paths = {
        'train': data_dir / 'train' / 'all_train.jsonl',
        'valid': data_dir / 'valid' / 'all_valid.jsonl', 
        'test': data_dir / 'test' / 'all_test.jsonl'
    }
    
    for split, filepath in split_paths.items():
        if filepath.exists():
            splits[split] = pd.read_json(filepath, lines=True)
            print(f"{split}: {len(splits[split])} Einträge geladen")
        else:
            print(f"Warnung: {filepath} nicht gefunden")
    
    return splits


def extract_texts_and_labels(df: pd.DataFrame) -> tuple:
    """
    Extrahiert Texte und Labels aus einem DataFrame.
    
    Args:
        df: DataFrame mit 'text_processed' und 'rating' Spalten
        
    Returns:
        Tuple aus (texts, labels)
    """
    texts = df['text_processed'].tolist()
    labels = df['rating'].values
    return texts, labels


def main():
    """
    Beispiel für die Verwendung der Vektorisierer.
    """
    print("🔧 Feature Engineering für Amazon Reviews Sentimentanalyse")
    print("=" * 60)
    
    # Daten laden
    data_dir = Path("data/processed")
    splits = load_processed_data(data_dir)
    
    if 'train' not in splits:
        print("❌ Trainingsdaten nicht gefunden!")
        return
    
    # Nur Trainingsdaten für Feature Engineering
    train_texts, train_labels = extract_texts_and_labels(splits['train'])
    
    print(f"\n📊 Trainingsdaten:")
    print(f"Trainingssamples: {len(train_texts)}")
    print(f"Label-Verteilung: {np.bincount(train_labels)}")
    
    # TF-IDF Vektorisierung
    print(f"\n🔤 TF-IDF Vektorisierung:")
    tfidf = TFIDFVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_texts)
    
    print(f"TF-IDF Features: {X_train_tfidf.shape[1]}")
    print(f"Top Features: {tfidf.get_top_features(10)}")
    
    # WordPiece Vektorisierung (Beispiel mit kleineren Daten)
    print(f"\n🔤 WordPiece Vektorisierung:")
    sample_size = min(1000, len(train_texts))  # Nur Sample für Demo
    wp = WordPieceVectorizer(max_length=128)
    X_train_wp = wp.fit_transform(train_texts[:sample_size])
    
    print(f"WordPiece Features: {X_train_wp.shape[1]}")
    print(f"Vokabulargröße: {wp.get_vocabulary_size()}")
    
    # Optional: Valid-Daten für Demonstration
    if 'valid' in splits:
        print(f"\n📊 Validierungsdaten (optional für Demo):")
        valid_texts, valid_labels = extract_texts_and_labels(splits['valid'])
        X_valid_tfidf = tfidf.transform(valid_texts)
        print(f"Valid TF-IDF Shape: {X_valid_tfidf.shape}")
    
    # Speichern der Vektorizer
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    tfidf.save(models_dir / "tfidf_vectorizer.pkl")
    wp.save(models_dir / "wordpiece_vectorizer.pkl")
    
    print(f"\n✅ Vektorizer gespeichert in {models_dir}")
    print(f"📁 TF-IDF: {models_dir / 'tfidf_vectorizer.pkl'}")
    print(f"📁 WordPiece: {models_dir / 'wordpiece_vectorizer.pkl'}")
    print(f"\n💡 Hinweis: Für das Training verwende nur die Train-Daten!")


if __name__ == "__main__":
    main()