#!/usr/bin/env python
"""
Test-Skript für die Vektorisierer.

Dieses Skript testet die TF-IDF und WordPiece Vektorisierer
mit den verarbeiteten Amazon Reviews Daten.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Vectorizer importieren
from vectorizer import TFIDFVectorizer, WordPieceVectorizer, load_processed_data, extract_texts_and_labels


def test_tfidf_vectorizer():
    """Testet den TF-IDF Vektorizer."""
    print("🧪 Teste TF-IDF Vektorizer...")
    
    # Kleine Testdaten
    test_texts = [
        "great product amazing quality",
        "terrible experience bad service",
        "good value for money",
        "disappointing purchase waste money",
        "excellent customer service"
    ]
    test_labels = [5, 1, 4, 2, 5]
    
    # TF-IDF Vektorizer erstellen und trainieren
    tfidf = TFIDFVectorizer(max_features=100, ngram_range=(1, 1))
    X = tfidf.fit_transform(test_texts)
    
    print(f"✅ TF-IDF Matrix Shape: {X.shape}")
    print(f"✅ Feature Names: {len(tfidf.get_feature_names())}")
    print(f"✅ Top Features: {tfidf.get_top_features(5)}")
    
    # Speichern und Laden testen
    test_path = Path("test_tfidf.pkl")
    tfidf.save(test_path)
    loaded_tfidf = TFIDFVectorizer.load(test_path)
    
    # Test mit geladenem Vektorizer
    X_loaded = loaded_tfidf.transform(test_texts)
    assert np.array_equal(X, X_loaded), "❌ Laden/Speichern funktioniert nicht!"
    print("✅ Speichern/Laden funktioniert")
    
    # Aufräumen
    test_path.unlink(missing_ok=True)
    return True


def test_wordpiece_vectorizer():
    """Testet den WordPiece Vektorizer."""
    print("\n🧪 Teste WordPiece Vektorizer...")
    
    # Kleine Testdaten
    test_texts = [
        "great product amazing quality",
        "terrible experience bad service",
        "good value for money"
    ]
    
    # WordPiece Vektorizer erstellen
    wp = WordPieceVectorizer(max_length=64)
    X = wp.fit_transform(test_texts)
    
    print(f"✅ WordPiece Matrix Shape: {X.shape}")
    print(f"✅ Vocabulary Size: {wp.get_vocabulary_size()}")
    print(f"✅ Feature Dimension: {wp.get_feature_dim()}")
    
    # Speichern und Laden testen
    test_path = Path("test_wordpiece.pkl")
    wp.save(test_path)
    loaded_wp = WordPieceVectorizer.load(test_path)
    
    # Test mit geladenem Vektorizer
    X_loaded = loaded_wp.transform(test_texts)
    assert np.array_equal(X, X_loaded), "❌ Laden/Speichern funktioniert nicht!"
    print("✅ Speichern/Laden funktioniert")
    
    # Aufräumen
    test_path.unlink(missing_ok=True)
    tokenizer_path = Path("test_wordpiece_tokenizer")
    if tokenizer_path.exists():
        import shutil
        shutil.rmtree(tokenizer_path)
    
    return True


def test_with_real_data():
    """Testet die Vektorizer mit echten Daten."""
    print("\n🧪 Teste mit echten Daten...")
    
    # Daten laden
    data_dir = Path("data/processed")
    splits = load_processed_data(data_dir)
    
    if 'train' not in splits:
        print("❌ Trainingsdaten nicht gefunden!")
        return False
    
    # Kleine Stichprobe für schnellen Test
    sample_df = splits['train'].sample(n=100, random_state=42)
    texts, labels = extract_texts_and_labels(sample_df)
    
    print(f"✅ Testdaten geladen: {len(texts)} Samples")
    print(f"✅ Label-Verteilung: {np.bincount(labels)}")
    
    # TF-IDF Test
    print("\n🔤 TF-IDF Test mit echten Daten:")
    tfidf = TFIDFVectorizer(max_features=1000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(texts)
    print(f"✅ TF-IDF Shape: {X_tfidf.shape}")
    print(f"✅ TF-IDF Top Features: {tfidf.get_top_features(10)}")
    
    # WordPiece Test (nur mit kleineren Daten)
    print("\n🔤 WordPiece Test mit echten Daten:")
    wp = WordPieceVectorizer(max_length=128)
    X_wp = wp.fit_transform(texts[:50])  # Nur 50 für schnellen Test
    print(f"✅ WordPiece Shape: {X_wp.shape}")
    print(f"✅ WordPiece Vocabulary: {wp.get_vocabulary_size()}")
    
    return True


def main():
    """Hauptfunktion für alle Tests."""
    print("🧪 Vectorizer Tests")
    print("=" * 50)
    
    try:
        # Einzelne Tests
        test_tfidf_vectorizer()
        test_wordpiece_vectorizer()
        test_with_real_data()
        
        print("\n✅ Alle Tests erfolgreich!")
        print("🎉 Feature Engineering ist bereit für das Training!")
        
    except Exception as e:
        print(f"\n❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)