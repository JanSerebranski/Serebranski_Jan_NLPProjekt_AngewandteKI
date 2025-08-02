#!/usr/bin/env python
"""
hyperparameter_optimization.py – Optuna-basierte Hyperparameter-Optimierung für BERT

Ziel: Optimierung der wichtigsten Hyperparameter für das BERT-Training
mit Fokus auf Accuracy und Trainingszeit.

Parameter: Learning Rate, Batch Size, Epochs, Max Length, Class Weight Method
Framework: Optuna für effiziente Suche
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Importiere BERT-Klassen aus train_hf.py
import sys
sys.path.append('.')
from train_hf import SentimentDataset, BERTSentimentTrainer


def load_optimization_data():
    """Lädt reduzierte Daten für Hyperparameter-Optimierung."""
    DATA_DIR = Path('data/processed')
    
    # Lade zusammengeführte Datensätze
    train_path = DATA_DIR / 'train' / 'all_train.jsonl'
    valid_path = DATA_DIR / 'valid' / 'all_valid.jsonl'
    
    if not all(p.exists() for p in [train_path, valid_path]):
        print("❌ Datensätze nicht gefunden!")
        return None, None
    
    # Lade Daten mit Chunking für Memory-Effizienz
    print("🔄 Lade Trainingsdaten...")
    train_chunks = []
    chunk_size = 10000
    
    for chunk in pd.read_json(train_path, lines=True, chunksize=chunk_size):
        train_chunks.append(chunk)
        if len(train_chunks) * chunk_size >= 5000:
            break
    
    train_df = pd.concat(train_chunks, ignore_index=True)
    train_df = train_df.sample(n=min(2000, len(train_df)), random_state=42)
    
    print("🔄 Lade Validierungsdaten...")
    valid_chunks = []
    
    for chunk in pd.read_json(valid_path, lines=True, chunksize=chunk_size):
        valid_chunks.append(chunk)
        if len(valid_chunks) * chunk_size >= 1000:
            break
    
    valid_df = pd.concat(valid_chunks, ignore_index=True)
    valid_df = valid_df.sample(n=min(500, len(valid_df)), random_state=42)
    
    print(f"✅ Optimierungsdaten geladen:")
    print(f"   Train: {len(train_df)} Samples")
    print(f"   Valid: {len(valid_df)} Samples")
    
    return train_df, valid_df


def objective(trial, train_df, valid_df, category='Automotive'):
    """Optuna Objective-Funktion für Hyperparameter-Optimierung."""
    
    # Hyperparameter-Suche
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
    epochs = trial.suggest_int('epochs', 1, 3)
    max_length = trial.suggest_categorical('max_length', [128, 256, 512])
    weight_method = trial.suggest_categorical('weight_method', ['balanced', 'none'])
    
    try:
        # Filtere Daten für Kategorie
        cat_train = train_df[train_df['category'] == category]
        cat_valid = valid_df[valid_df['category'] == category]
        
        if len(cat_train) < 50 or len(cat_valid) < 10:
            return 0.0  # Zu wenig Daten
        
        # Erstelle Trainer
        trainer = BERTSentimentTrainer()
        trainer.setup_model()
        
        # Class Weights
        if weight_method == 'balanced':
            class_weights = trainer.calculate_class_weights(cat_train['rating'].values)
        else:
            class_weights = None
        
        # Erstelle DataLoaders
        train_loader, valid_loader = trainer.create_data_loaders(
            cat_train['text'].values,
            cat_train['rating'].values - 1,
            cat_valid['text'].values,
            cat_valid['rating'].values - 1,
            batch_size=batch_size,
            max_length=max_length
        )
        
        # Training
        training_results = trainer.train(
            train_loader, valid_loader, 
            epochs=epochs, 
            learning_rate=learning_rate
        )
        
        # Beste Accuracy als Objective
        best_accuracy = training_results['best_accuracy']
        
        # Berichte Zwischenergebnisse
        trial.report(best_accuracy)
        
        return best_accuracy
        
    except Exception as e:
        print(f"❌ Fehler in Trial: {e}")
        return 0.0


def run_hyperparameter_optimization():
    """Führt Hyperparameter-Optimierung durch."""
    print("🔍 Starte Hyperparameter-Optimierung...")
    
    # Lade Daten
    train_df, valid_df = load_optimization_data()
    if train_df is None:
        return
    
    # Optuna Study erstellen
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimierung für jede Kategorie
    categories = ['Automotive', 'Books', 'Video_Games']
    best_params = {}
    
    for category in categories:
        print(f"\n📊 Optimierung für Kategorie: {category}")
        
        # Erstelle Study für diese Kategorie
        study_name = f"bert_optimization_{category.lower()}"
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Führe Optimierung durch
        study.optimize(
            lambda trial: objective(trial, train_df, valid_df, category),
            n_trials=10,  # Reduziert auf 10 Trials für schnellere Optimierung
            timeout=1800   # 30 Minuten Timeout
        )
        
        # Beste Parameter speichern
        best_params[category] = {
            'best_accuracy': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }
        
        print(f"✅ {category} Optimierung abgeschlossen:")
        print(f"   Beste Accuracy: {study.best_value:.4f}")
        print(f"   Beste Parameter: {study.best_params}")
    
    # Speichere Ergebnisse
    save_optimization_results(best_params)
    
    return best_params


def save_optimization_results(best_params):
    """Speichert Optimierungsergebnisse."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'optimization_method': 'Optuna TPE',
        'n_trials_per_category': 20,
        'categories': list(best_params.keys()),
        'best_parameters': best_params
    }
    
    # Speichern
    results_path = Path('results/hyperparameter_optimization.json')
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Optimierungsergebnisse gespeichert: {results_path}")


def print_optimization_summary(best_params):
    """Zeigt Zusammenfassung der Optimierungsergebnisse."""
    print("\n" + "="*60)
    print("🎯 HYPERPARAMETER-OPTIMIERUNG ZUSAMMENFASSUNG")
    print("="*60)
    
    for category, results in best_params.items():
        print(f"\n📊 {category}:")
        print(f"   Beste Accuracy: {results['best_accuracy']:.4f}")
        print(f"   Trials: {results['n_trials']}")
        print(f"   Parameter:")
        for param, value in results['best_params'].items():
            print(f"     {param}: {value}")
    
    # Durchschnittliche Accuracy
    avg_accuracy = np.mean([r['best_accuracy'] for r in best_params.values()])
    print(f"\n📈 Durchschnittliche Accuracy: {avg_accuracy:.4f}")
    
    # Empfohlene Parameter für Training
    print(f"\n🚀 EMPFOHLENE PARAMETER FÜR TRAINING:")
    print(f"   Learning Rate: 2e-5 bis 5e-5")
    print(f"   Batch Size: 4-8")
    print(f"   Epochs: 1-2")
    print(f"   Max Length: 256")
    print(f"   Class Weights: balanced")


def main():
    """Hauptfunktion für Hyperparameter-Optimierung."""
    print("🔍 Starte Hyperparameter-Optimierung für BERT...")
    
    # Führe Optimierung durch
    best_params = run_hyperparameter_optimization()
    
    if best_params:
        # Zeige Zusammenfassung
        print_optimization_summary(best_params)
        
        print(f"\n✅ Hyperparameter-Optimierung abgeschlossen!")
        print(f"📋 Nächste Schritte:")
        print(f"   1. Überprüfe Ergebnisse in results/hyperparameter_optimization.json")
        print(f"   2. Passe train_hf.py mit besten Parametern an")
        print(f"   3. Starte vollständiges Training")
    else:
        print("❌ Optimierung fehlgeschlagen!")


if __name__ == "__main__":
    main() 