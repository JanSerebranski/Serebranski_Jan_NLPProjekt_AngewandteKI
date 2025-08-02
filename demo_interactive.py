#!/usr/bin/env python
"""
demo_interactive.py – Interaktives Demo für Sentimentanalyse

Ziel: Benutzer können fiktive Amazon-Rezensionen eingeben und 
die Sternebewertung (1-5) durch beide Modelle vorhersagen lassen.
"""

import pickle
import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

class SentimentAnalyzer:
    def __init__(self):
        """Initialisiert beide Modelle (Baseline und BERT)."""
        self.baseline_models = {}
        self.bert_models = {}
        self.tokenizers = {}
        self.categories = ["Automotive", "Books", "Video_Games"]
        
        print("🔧 Lade Modelle...")
        self._load_models()
        print("✅ Modelle erfolgreich geladen!")
    
    def _load_models(self):
        """Lädt alle trainierten Modelle."""
        # Baseline Modelle laden
        for category in self.categories:
            baseline_path = f"models/baseline_{category.lower()}.pkl"
            if os.path.exists(baseline_path):
                with open(baseline_path, 'rb') as f:
                    self.baseline_models[category] = pickle.load(f)
        
        # BERT Modelle laden
        for category in self.categories:
            bert_path = f"models/bert_{category.lower()}"
            if os.path.exists(f"{bert_path}/model.safetensors"):
                self.tokenizers[category] = DistilBertTokenizer.from_pretrained(bert_path)
                self.bert_models[category] = DistilBertForSequenceClassification.from_pretrained(bert_path)
                self.bert_models[category].eval()
    
    def preprocess_text(self, text):
        """Bereitet Text für Baseline-Modell vor."""
        # Einfache Bereinigung
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # HTML-Tags entfernen
        text = re.sub(r'[^\w\s]', '', text)   # Sonderzeichen entfernen
        return text
    
    def predict_baseline(self, text, category):
        """Vorhersage mit Baseline-Modell."""
        if category not in self.baseline_models:
            return None, 0.0
        
        # Text vorbereiten
        processed_text = self.preprocess_text(text)
        
        # Vorhersage
        try:
            prediction = self.baseline_models[category].predict([processed_text])[0]
            confidence = np.max(self.baseline_models[category].predict_proba([processed_text]))
            return prediction, confidence
        except:
            return None, 0.0
    
    def predict_bert(self, text, category):
        """Vorhersage mit BERT-Modell."""
        if category not in self.bert_models:
            return None, 0.0
        
        try:
            # Tokenisierung
            inputs = self.tokenizers[category](
                text, 
                truncation=True, 
                padding=True, 
                max_length=256, 
                return_tensors="pt"
            )
            
            # Vorhersage
            with torch.no_grad():
                outputs = self.bert_models[category](**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            return prediction + 1, confidence  # +1 weil BERT 0-4, aber wir wollen 1-5
        except:
            return None, 0.0
    
    def analyze_review(self, text, category):
        """Analysiert eine Rezension mit beiden Modellen."""
        print(f"\n📝 Analysiere Rezension für Kategorie: {category}")
        print(f"Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print("-" * 60)
        
        # Baseline Vorhersage
        baseline_pred, baseline_conf = self.predict_baseline(text, category)
        if baseline_pred is not None:
            print(f"🔸 Baseline-Modell: {baseline_pred} Sterne (Confidence: {baseline_conf:.2%})")
        else:
            print("🔸 Baseline-Modell: Fehler bei Vorhersage")
        
        # BERT Vorhersage
        bert_pred, bert_conf = self.predict_bert(text, category)
        if bert_pred is not None:
            print(f"🤖 BERT-Modell: {bert_pred} Sterne (Confidence: {bert_conf:.2%})")
        else:
            print("🤖 BERT-Modell: Fehler bei Vorhersage")
        
        # Vergleich
        if baseline_pred is not None and bert_pred is not None:
            if baseline_pred == bert_pred:
                print(f"✅ Beide Modelle stimmen überein: {baseline_pred} Sterne")
            else:
                print(f"⚠️ Modelle unterscheiden sich: Baseline={baseline_pred}, BERT={bert_pred}")
        
        print("-" * 60)
        return baseline_pred, bert_pred

def get_star_rating_description(rating):
    """Gibt eine Beschreibung der Sternebewertung zurück."""
    descriptions = {
        1: "Sehr schlecht - Nicht empfehlenswert",
        2: "Schlecht - Nicht zufriedenstellend", 
        3: "Durchschnittlich - Mittelmäßig",
        4: "Gut - Empfehlenswert",
        5: "Sehr gut - Hervorragend"
    }
    return descriptions.get(rating, "Unbekannt")

def main():
    """Hauptfunktion für interaktives Demo."""
    print("🌟 Amazon Rezension Sentiment-Analyzer Demo")
    print("=" * 60)
    print("Gib fiktive Amazon-Rezensionen ein und lass die Sternebewertung vorhersagen!")
    print("Verfügbare Kategorien: Automotive, Books, Video_Games")
    print("Beende mit 'quit' oder 'exit'")
    print("=" * 60)
    
    # Modelle laden
    analyzer = SentimentAnalyzer()
    
    # Beispiel-Rezensionen
    examples = {
        "Automotive": [
            "Das Auto ist fantastisch! Tolle Beschleunigung und sehr sparsam im Verbrauch.",
            "Schreckliche Qualität. Nach einem Monat schon kaputt. Nie wieder!",
            "Ganz okay, aber nichts Besonderes. Preis-Leistung ist in Ordnung."
        ],
        "Books": [
            "Ein absolutes Meisterwerk! Spannend von der ersten bis zur letzten Seite.",
            "Langweilig und schlecht geschrieben. Zeitverschwendung.",
            "Interessant, aber manchmal etwas verwirrend. Insgesamt lesenswert."
        ],
        "Video_Games": [
            "Das beste Spiel aller Zeiten! Grafik und Gameplay sind perfekt.",
            "Vollständiger Reinfall. Bugs überall und schlechte Grafik.",
            "Ganz nett, aber nichts Neues. Für den Preis akzeptabel."
        ]
    }
    
    while True:
        print("\n🎮 Was möchtest du tun?")
        print("1. Eigene Rezension eingeben")
        print("2. Beispiel-Rezensionen testen")
        print("3. Beenden")
        
        choice = input("\nDeine Wahl (1-3): ").strip()
        
        if choice == "1":
            # Eigene Rezension
            print("\n📝 Gib deine Rezension ein:")
            text = input("Rezension: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                print("❌ Bitte gib eine Rezension ein!")
                continue
            
            print("\n🏷️ Wähle eine Kategorie:")
            for i, cat in enumerate(analyzer.categories, 1):
                print(f"{i}. {cat}")
            
            try:
                cat_choice = int(input("Kategorie (1-3): ")) - 1
                if 0 <= cat_choice < len(analyzer.categories):
                    category = analyzer.categories[cat_choice]
                    analyzer.analyze_review(text, category)
                else:
                    print("❌ Ungültige Kategorie!")
            except ValueError:
                print("❌ Bitte gib eine Zahl ein!")
        
        elif choice == "2":
            # Beispiel-Rezensionen
            print("\n📚 Beispiel-Rezensionen testen:")
            for category, reviews in examples.items():
                print(f"\n--- {category} ---")
                for i, review in enumerate(reviews, 1):
                    print(f"\nBeispiel {i}:")
                    analyzer.analyze_review(review, category)
                    input("Drücke Enter für nächstes Beispiel...")
        
        elif choice == "3":
            break
        
        else:
            print("❌ Ungültige Wahl!")
    
    print("\n👋 Danke für das Testen! Auf Wiedersehen!")

if __name__ == "__main__":
    main() 