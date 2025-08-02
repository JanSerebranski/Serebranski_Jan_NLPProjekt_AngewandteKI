#!/usr/bin/env python
"""
quick_test.py – Schneller Test für einzelne Rezensionen

Einfache Version zum schnellen Testen von Rezensionen.
"""

import pickle
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import re
import os

def load_models(category):
    """Lädt Modelle für eine Kategorie."""
    models = {}
    
    # Baseline laden
    baseline_path = f"models/baseline_{category.lower()}.pkl"
    if os.path.exists(baseline_path):
        with open(baseline_path, 'rb') as f:
            models['baseline'] = pickle.load(f)
    
    # BERT laden
    bert_path = f"models/bert_{category.lower()}"
    if os.path.exists(f"{bert_path}/model.safetensors"):
        models['tokenizer'] = DistilBertTokenizer.from_pretrained(bert_path)
        models['bert'] = DistilBertForSequenceClassification.from_pretrained(bert_path)
        models['bert'].eval()
    
    return models

def predict_review(text, category):
    """Vorhersage für eine einzelne Rezension."""
    print(f"\n🔍 Analysiere: '{text}'")
    print(f"📂 Kategorie: {category}")
    print("-" * 50)
    
    models = load_models(category)
    
    # Baseline Vorhersage
    if 'baseline' in models:
        try:
            # Text vorbereiten
            processed_text = text.lower()
            processed_text = re.sub(r'<[^>]+>', '', processed_text)
            processed_text = re.sub(r'[^\w\s]', '', processed_text)
            
            # Vorhersage
            prediction = models['baseline'].predict([processed_text])[0]
            confidence = np.max(models['baseline'].predict_proba([processed_text]))
            
            print(f"🔸 Baseline: {prediction} Sterne ({confidence:.1%})")
        except Exception as e:
            print(f"🔸 Baseline: Fehler - {e}")
    else:
        print("🔸 Baseline: Modell nicht gefunden")
    
    # BERT Vorhersage
    if 'bert' in models:
        try:
            # Tokenisierung
            inputs = models['tokenizer'](
                text, 
                truncation=True, 
                padding=True, 
                max_length=256, 
                return_tensors="pt"
            )
            
            # Vorhersage
            with torch.no_grad():
                outputs = models['bert'](**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item() + 1
                confidence = probabilities[0][prediction-1].item()
            
            print(f"🤖 BERT: {prediction} Sterne ({confidence:.1%})")
        except Exception as e:
            print(f"🤖 BERT: Fehler - {e}")
    else:
        print("🤖 BERT: Modell nicht gefunden")
    
    print("-" * 50)

def main():
    """Hauptfunktion."""
    print("⭐ Quick Test - Amazon Rezension Sentiment-Analyzer")
    print("=" * 60)
    
    # Verfügbare Kategorien
    categories = ["Automotive", "Books", "Video_Games"]
    
    while True:
        print("\n📝 Gib eine Rezension ein (oder 'quit' zum Beenden):")
        text = input("> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            print("❌ Bitte gib eine Rezension ein!")
            continue
        
        print("\n🏷️ Wähle Kategorie:")
        for i, cat in enumerate(categories, 1):
            print(f"{i}. {cat}")
        
        try:
            choice = int(input("Kategorie (1-3): ")) - 1
            if 0 <= choice < len(categories):
                predict_review(text, categories[choice])
            else:
                print("❌ Ungültige Kategorie!")
        except ValueError:
            print("❌ Bitte gib eine Zahl ein!")
    
    print("\n👋 Auf Wiedersehen!")

if __name__ == "__main__":
    main() 