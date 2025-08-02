#!/usr/bin/env python
"""
baseline_training.py – Naive Bayes Baseline-Modell Training

Ziel: Training und Evaluation eines Multinomial Naive Bayes Modells als Baseline 
für die Sentimentanalyse von Amazon-Produktrezensionen.

Modell: Multinomial Naive Bayes mit TF-IDF Vektorisierung
Metriken: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
Datensätze: Automotive, Books, Video_Games (Train/Valid/Test Splits)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# Styling für bessere Visualisierungen
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Setup abgeschlossen")


class SentimentPipeline:
    """Vollständige Pipeline für Sentimentanalyse mit bereits preprocessed Daten."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline = None
        self._create_pipeline()
    
    def _create_pipeline(self):
        """Erstellt die scikit-learn Pipeline für bereits preprocessed Daten."""
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )),
            ('classifier', MultinomialNB())
        ])
    
    def fit(self, X, y):
        """Trainiert die Pipeline."""
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """Macht Vorhersagen."""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Gibt Wahrscheinlichkeiten zurück."""
        return self.pipeline.predict_proba(X)
    
    def score(self, X, y):
        """Berechnet die Accuracy."""
        return self.pipeline.score(X, y)
    
    def save(self, filepath):
        """Speichert die Pipeline."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    @classmethod
    def load(cls, filepath):
        """Lädt eine gespeicherte Pipeline."""
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        instance = cls()
        instance.pipeline = pipeline
        return instance


def load_data():
    """Lädt die zusammengeführten preprocessed Datensätze."""
    DATA_DIR = Path('data/processed')
    
    # Lade zusammengeführte Datensätze
    train_path = DATA_DIR / 'train' / 'all_train.jsonl'
    valid_path = DATA_DIR / 'valid' / 'all_valid.jsonl'
    test_path = DATA_DIR / 'test' / 'all_test.jsonl'
    
    print("Lade zusammengeführte Datensätze...")
    train_df = pd.read_json(train_path, lines=True)
    valid_df = pd.read_json(valid_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    print(f"Trainingsdaten: {len(train_df)} Einträge")
    print(f"Validierungsdaten: {len(valid_df)} Einträge")
    print(f"Testdaten: {len(test_df)} Einträge")
    
    # Kategorien aus den Daten extrahieren
    categories = sorted(train_df['category'].unique())
    print(f"Gefundene Kategorien: {categories}")
    
    # Daten nach Kategorien aufteilen
    data = {}
    for cat in categories:
        data[cat] = {
            'train': train_df[train_df['category'] == cat],
            'valid': valid_df[valid_df['category'] == cat],
            'test': test_df[test_df['category'] == cat]
        }
        print(f"{cat}: {len(data[cat]['train'])} train, {len(data[cat]['valid'])} valid, {len(data[cat]['test'])} test")
    
    print("\n✅ Alle zusammengeführten Datensätze geladen")
    return data, categories


def train_baseline_models(data, categories):
    """Trainiert Baseline-Modelle für alle Kategorien."""
    results = {}
    
    for cat in categories:
        print(f"\n{'='*50}")
        print(f"Training für Kategorie: {cat}")
        print(f"{'='*50}")
        
        # Daten vorbereiten (bereits preprocessed)
        train_texts = data[cat]['train']['text'].tolist()
        train_labels = data[cat]['train']['rating'].values
        valid_texts = data[cat]['valid']['text'].tolist()
        valid_labels = data[cat]['valid']['rating'].values
        test_texts = data[cat]['test']['text'].tolist()
        test_labels = data[cat]['test']['rating'].values
        
        print(f"Trainingsdaten: {len(train_texts)} Texte (bereits preprocessed)")
        print(f"Validierungsdaten: {len(valid_texts)} Texte (bereits preprocessed)")
        print(f"Testdaten: {len(test_texts)} Texte (bereits preprocessed)")
        
        # Pipeline erstellen und trainieren
        pipeline = SentimentPipeline()
        print("\nTraining läuft...")
        pipeline.fit(train_texts, train_labels)
        
        # Vorhersagen machen
        train_pred = pipeline.predict(train_texts)
        valid_pred = pipeline.predict(valid_texts)
        test_pred = pipeline.predict(test_texts)
        
        # Metriken berechnen
        train_acc = accuracy_score(train_labels, train_pred)
        valid_acc = accuracy_score(valid_labels, valid_pred)
        test_acc = accuracy_score(test_labels, test_pred)
        
        # Detaillierte Metriken für Test-Set
        test_report = classification_report(test_labels, test_pred, output_dict=True)
        test_conf_matrix = confusion_matrix(test_labels, test_pred)
        
        # Ergebnisse speichern
        results[cat] = {
            'valid_accuracy': valid_acc,
            'test_accuracy': test_acc,
            'test_predictions': test_pred,
            'test_labels': test_labels,
            'classification_report': test_report,
            'confusion_matrix': test_conf_matrix,
            'pipeline': pipeline
        }
        
        print(f"\nErgebnisse für {cat}:")
        print(f"Valid Accuracy: {valid_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Train Accuracy (nur Debug): {train_acc:.4f}")
        
        # Modell speichern
        model_path = Path(f'models/baseline_{cat.lower()}.pkl')
        model_path.parent.mkdir(exist_ok=True)
        pipeline.save(model_path)
        print(f"Modell gespeichert: {model_path}")
    
    print("\n✅ Training für alle Kategorien abgeschlossen")
    return results


def visualize_results(results, categories):
    """Visualisiert die Ergebnisse."""
    # Accuracy Vergleich
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy pro Kategorie (nur Valid und Test)
    valid_accs = [results[cat]['valid_accuracy'] for cat in categories]
    test_accs = [results[cat]['test_accuracy'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, valid_accs, width, label='Valid', alpha=0.8, color='#f39c12')
    ax1.bar(x + width/2, test_accs, width, label='Test', alpha=0.8, color='#e74c3c')
    
    ax1.set_xlabel('Kategorien')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Naive Bayes Baseline - Valid/Test Accuracy pro Kategorie')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Werte auf Balken anzeigen
    for i, (valid, test) in enumerate(zip(valid_accs, test_accs)):
        ax1.text(i - width/2, valid + 0.01, f'{valid:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom')
    
    # Gesamtübersicht
    overall_valid = np.mean(valid_accs)
    overall_test = np.mean(test_accs)
    
    ax2.bar(['Valid', 'Test'], [overall_valid, overall_test], 
            color=['#f39c12', '#e74c3c'], alpha=0.8)
    ax2.set_ylabel('Durchschnittliche Accuracy')
    ax2.set_title('Gesamtübersicht - Naive Bayes Baseline')
    ax2.grid(True, alpha=0.3)
    
    # Werte auf Balken anzeigen
    for i, (label, value) in enumerate(zip(['Valid', 'Test'], 
                                          [overall_valid, overall_test])):
        ax2.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/baseline_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nGesamtübersicht:")
    print(f"Durchschnittliche Valid Accuracy: {overall_valid:.4f}")
    print(f"Durchschnittliche Test Accuracy: {overall_test:.4f}")
    
    return overall_test


def create_confusion_matrices(results, categories):
    """Erstellt Confusion Matrices für alle Kategorien."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, cat in enumerate(categories):
        conf_matrix = results[cat]['confusion_matrix']
        
        # Normalisierte Confusion Matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        im = axes[i].imshow(conf_matrix_norm, cmap='Blues', aspect='auto')
        axes[i].set_title(f'{cat} - Confusion Matrix (Test)')
        axes[i].set_xlabel('Vorhergesagtes Rating')
        axes[i].set_ylabel('Tatsächliches Rating')
        
        # Werte in Zellen anzeigen
        for j in range(5):
            for k in range(5):
                text = axes[i].text(k, j, f'{conf_matrix_norm[j, k]:.2f}',
                                   ha='center', va='center', 
                                   color='white' if conf_matrix_norm[j, k] > 0.5 else 'black')
        
        axes[i].set_xticks(range(5))
        axes[i].set_yticks(range(5))
        axes[i].set_xticklabels(range(1, 6))
        axes[i].set_yticklabels(range(1, 6))
    
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.savefig('results/baseline_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_detailed_metrics(results, categories):
    """Gibt detaillierte Metriken aus."""
    for cat in categories:
        print(f"\n{'='*60}")
        print(f"Detaillierte Metriken für {cat}")
        print(f"{'='*60}")
        
        report = results[cat]['classification_report']
        
        # Accuracy
        print(f"\nAccuracy: {results[cat]['test_accuracy']:.4f}")
        
        # Per-Klasse Metriken
        print("\nPer-Klasse Metriken:")
        print("Rating | Precision | Recall | F1-Score | Support")
        print("-" * 50)
        
        for rating in range(1, 6):
            if str(rating) in report:
                precision = report[str(rating)]['precision']
                recall = report[str(rating)]['recall']
                f1 = report[str(rating)]['f1-score']
                support = report[str(rating)]['support']
                
                print(f"{rating:6d} | {precision:9.3f} | {recall:6.3f} | {f1:9.3f} | {support:7.0f}")
        
        # Macro und Weighted Averages
        print(f"\nMacro Average:")
        print(f"Precision: {report['macro avg']['precision']:.4f}")
        print(f"Recall: {report['macro avg']['recall']:.4f}")
        print(f"F1-Score: {report['macro avg']['f1-score']:.4f}")
        
        print(f"\nWeighted Average:")
        print(f"Precision: {report['weighted avg']['precision']:.4f}")
        print(f"Recall: {report['weighted avg']['recall']:.4f}")
        print(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")


def save_results(results, overall_test, categories):
    """Speichert die Ergebnisse."""
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Multinomial Naive Bayes',
        'vectorizer': 'TF-IDF',
        'categories': categories,
        'overall_accuracy': overall_test,
        'category_results': {}
    }
    
    for cat in categories:
        results_summary['category_results'][cat] = {
            'valid_accuracy': results[cat]['valid_accuracy'],
            'test_accuracy': results[cat]['test_accuracy'],
            'classification_report': results[cat]['classification_report']
        }
    
    # Speichern
    results_path = Path('results/baseline_results.json')
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"Ergebnisse gespeichert: {results_path}")


def print_summary(results, overall_test, categories, data):
    """Gibt eine Zusammenfassung aus."""
    print("="*80)
    print("ZUSAMMENFASSUNG: Naive Bayes Baseline-Modell")
    print("="*80)
    
    # Gesamtstatistiken
    print(f"\nGesamtübersicht:")
    print(f"Anzahl Kategorien: {len(categories)}")
    print(f"Durchschnittliche Test Accuracy: {overall_test:.4f}")
    print(f"Beste Kategorie: {max(results.keys(), key=lambda x: results[x]['test_accuracy'])} "
          f"({max(results.values(), key=lambda x: x['test_accuracy'])['test_accuracy']:.4f})")
    print(f"Schlechteste Kategorie: {min(results.keys(), key=lambda x: results[x]['test_accuracy'])} "
          f"({min(results.values(), key=lambda x: x['test_accuracy'])['test_accuracy']:.4f})")
    
    # Vergleich mit Zielvorgaben
    print(f"\nVergleich mit Projektzielen:")
    print(f"Ziel: Accuracy ≥ 70%")
    print(f"Erreicht: {overall_test*100:.1f}%")
    print(f"Status: {'✅ ERREICHT' if overall_test >= 0.7 else '❌ NICHT ERREICHT'}")
    
    # Mehrheitsklasse Vergleich
    print(f"\nMehrheitsklasse Analyse:")
    for cat in categories:
        train_labels = data[cat]['train']['rating'].values
        majority_class = pd.Series(train_labels).mode()[0]
        majority_accuracy = (train_labels == majority_class).mean()
        baseline_accuracy = results[cat]['test_accuracy']
        improvement = baseline_accuracy - majority_accuracy
        
        print(f"{cat}: Baseline {baseline_accuracy:.4f} vs Mehrheitsklasse {majority_accuracy:.4f} "
              f"(Verbesserung: {improvement:.4f})")
    
    # Empfehlungen
    print(f"\nEmpfehlungen für nächstes Modell:")
    if overall_test < 0.7:
        print("• Höhere Komplexität: BERT oder SVM mit RBF-Kernel")
        print("• Feature Engineering: N-Grams, Sentiment Lexicons")
        print("• Hyperparameter Tuning: Grid Search für TF-IDF Parameter")
    else:
        print("• Baseline erfüllt Ziele - kann als Referenz dienen")
        print("• Weiterer Verbesserung durch fortgeschrittene Modelle möglich")
    
    print(f"\n✅ Baseline-Modell Training und Evaluation abgeschlossen")


def main():
    """Hauptfunktion."""
    print("🚀 Starte Naive Bayes Baseline Training...")
    
    # Daten laden
    data, categories = load_data()
    
    # Modelle trainieren
    results = train_baseline_models(data, categories)
    
    # Ergebnisse visualisieren
    overall_test = visualize_results(results, categories)
    
    # Confusion Matrices erstellen
    create_confusion_matrices(results, categories)
    
    # Detaillierte Metriken ausgeben
    print_detailed_metrics(results, categories)
    
    # Ergebnisse speichern
    save_results(results, overall_test, categories)
    
    # Zusammenfassung ausgeben
    print_summary(results, overall_test, categories, data)


if __name__ == "__main__":
    main()