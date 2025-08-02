#!/usr/bin/env python
"""
train_hf.py – BERT/DistilBERT Fine-Tuning für Sentimentanalyse

Ziel: Training eines fortgeschrittenen Transformer-Modells zur Verbesserung
der Baseline-Ergebnisse mit Fokus auf Klassenungleichgewicht.

Modell: DistilBERT mit Class Weights
Metriken: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
Datensätze: Automotive, Books, Video_Games (Train/Valid/Test Splits)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import warnings
import pickle
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Styling für bessere Visualisierungen
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ BERT Training Setup abgeschlossen")


class SentimentDataset(Dataset):
    """Custom Dataset für Sentimentanalyse mit BERT."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Kürzere Sequenzen
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenisierung
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTSentimentTrainer:
    """BERT Trainer für Sentimentanalyse mit Class Weights."""
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=5, 
                 device=None, random_state=42):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        
        # Setze Seeds für Reproduzierbarkeit
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.tokenizer = None
        self.model = None
        self.class_weights = None
        
        print(f"✅ BERT Trainer initialisiert auf {self.device}")
    
    def setup_model(self):
        """Initialisiert Tokenizer und Modell."""
        print(f"🔄 Lade {self.model_name}...")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        self.model.to(self.device)
        print(f"✅ Modell geladen: {self.model_name}")
    
    def calculate_class_weights(self, labels):
        """Berechnet Class Weights für unausgeglichene Klassen."""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Berechne Class Weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        print("📊 Class Weights berechnet:")
        for i, weight in enumerate(class_weights):
            print(f"   Rating {i+1}: {weight:.3f}")
        
        return self.class_weights
    
    def create_data_loaders(self, train_texts, train_labels, valid_texts, valid_labels,
                           batch_size=16, max_length=256):  # Kürzere Sequenzen
        """Erstellt DataLoader für Training und Validation."""
        
        # Erstelle Datasets
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, max_length)
        valid_dataset = SentimentDataset(valid_texts, valid_labels, self.tokenizer, max_length)
        
        # Erstelle DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Für Windows Kompatibilität
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"✅ DataLoaders erstellt: Train={len(train_loader)}, Valid={len(valid_loader)}")
        return train_loader, valid_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler, criterion):
        """Trainiert eine Epoch."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        from tqdm import tqdm
        
        # Fortschrittsanzeige
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Metriken
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += len(labels)
            
            # Update Progress Bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = correct_predictions.double() / total_predictions
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        return total_loss / len(train_loader), correct_predictions.double() / total_predictions
    
    def eval_epoch(self, valid_loader, criterion):
        """Evaluates eine Epoch."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_preds = []
        all_labels = []
        
        from tqdm import tqdm
        
        # Fortschrittsanzeige für Evaluation
        progress_bar = tqdm(valid_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Metriken
                total_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += len(labels)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return (total_loss / len(valid_loader), 
                correct_predictions.double() / total_predictions,
                all_preds, all_labels)
    
    def train(self, train_loader, valid_loader, epochs=3, learning_rate=2e-5):
        """Trainiert das Modell."""
        print(f"🚀 Starte Training: {epochs} Epochs, LR={learning_rate}")
        
        # Optimizer und Scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Loss function mit Class Weights
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Training Loop
        best_accuracy = 0
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler, criterion)
            
            # Validation
            valid_loss, valid_acc, preds, labels = self.eval_epoch(valid_loader, criterion)
            
            # Speichere Metriken
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_accuracies.append(train_acc.item())
            valid_accuracies.append(valid_acc.item())
            
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"   Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
            
            # Best Model speichern
            if valid_acc > best_accuracy:
                best_accuracy = valid_acc
                best_model_state = self.model.state_dict().copy()
                print(f"   🎯 Neuer Best Score: {best_accuracy:.4f}")
        
        # Best Model wiederherstellen
        self.model.load_state_dict(best_model_state)
        
        # Finale Evaluation
        _, _, final_preds, final_labels = self.eval_epoch(valid_loader, criterion)
        
        results = {
            'best_accuracy': best_accuracy.item(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_accuracies': train_accuracies,
            'valid_accuracies': valid_accuracies,
            'final_predictions': final_preds,
            'final_labels': final_labels
        }
        
        print(f"\n✅ Training abgeschlossen! Best Accuracy: {best_accuracy:.4f}")
        return results
    
    def predict(self, texts, batch_size=16):
        """Macht Vorhersagen für neue Texte."""
        self.model.eval()
        predictions = []
        
        # Erstelle Dataset
        dataset = SentimentDataset(texts, [0] * len(texts), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return predictions
    
    def save_model(self, filepath):
        """Speichert das trainierte Modell."""
        model_dir = Path(filepath).parent
        model_dir.mkdir(exist_ok=True)
        
        # Speichere Modell und Tokenizer
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        print(f"💾 Modell gespeichert: {filepath}")
    
    @classmethod
    def load_model(cls, filepath, device=None):
        """Lädt ein trainiertes Modell."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tokenizer = DistilBertTokenizer.from_pretrained(filepath)
        model = DistilBertForSequenceClassification.from_pretrained(filepath)
        model.to(device)
        
        instance = cls(device=device)
        instance.tokenizer = tokenizer
        instance.model = model
        
        print(f"✅ Modell geladen: {filepath}")
        return instance


def load_data_for_bert():
    """Lädt Daten für BERT Training."""
    DATA_DIR = Path('data/processed')
    
    # Lade zusammengeführte Datensätze
    train_path = DATA_DIR / 'train' / 'all_train.jsonl'
    valid_path = DATA_DIR / 'valid' / 'all_valid.jsonl'
    test_path = DATA_DIR / 'test' / 'all_test.jsonl'
    
    if not all(p.exists() for p in [train_path, valid_path, test_path]):
        print("❌ Datensätze nicht gefunden!")
        return None, None, None
    
    # Lade Daten
    train_df = pd.read_json(train_path, lines=True)
    valid_df = pd.read_json(valid_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    print(f"✅ Daten geladen:")
    print(f"   Train: {len(train_df)} Samples")
    print(f"   Valid: {len(valid_df)} Samples")
    print(f"   Test: {len(test_df)} Samples")
    
    return train_df, valid_df, test_df


def load_optimized_hyperparameters():
    """Lädt optimierte Hyperparameter aus der Optimierung."""
    results_path = Path('results/hyperparameter_optimization.json')
    
    if not results_path.exists():
        print("⚠️ Keine optimierten Hyperparameter gefunden. Verwende Standard-Parameter.")
        return None
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            optimization_results = json.load(f)
        
        print("✅ Optimierte Hyperparameter geladen!")
        return optimization_results['best_parameters']
    except Exception as e:
        print(f"❌ Fehler beim Laden der Hyperparameter: {e}")
        return None


def train_bert_models(data, categories):
    """Trainiert BERT Modelle für alle Kategorien mit optimierten Hyperparametern."""
    print(f"\n🚀 Starte BERT Training für {len(categories)} Kategorien...")
    
    # Lade optimierte Hyperparameter
    optimized_params = load_optimized_hyperparameters()
    
    results = {}
    models = {}
    
    # ERHÖHE DATENMENGE für bessere Ergebnisse (6-8 Stunden Training)
    # Verwende den zusammengeführten Trainingssatz mit mehr Daten
    all_train = data['train'].sample(n=min(100000, len(data['train'])), random_state=42)
    all_valid = data['valid'].sample(n=min(20000, len(data['valid'])), random_state=42)
    all_test = data['test'].sample(n=min(20000, len(data['test'])), random_state=42)
    
    print(f"📊 Reduzierte Gesamtdaten:")
    print(f"   Train: {len(all_train)} Samples (aus allen Kategorien)")
    print(f"   Valid: {len(all_valid)} Samples (aus allen Kategorien)")
    print(f"   Test: {len(all_test)} Samples (aus allen Kategorien)")
    
    for cat in categories:
        print(f"\n📊 Training für Kategorie: {cat}")
        
        # Filtere Daten für Kategorie aus den reduzierten Gesamtdaten
        cat_train = all_train[all_train['category'] == cat]
        cat_valid = all_valid[all_valid['category'] == cat]
        cat_test = all_test[all_test['category'] == cat]
        
        print(f"   Train: {len(cat_train)} Samples für {cat}")
        print(f"   Valid: {len(cat_valid)} Samples für {cat}")
        print(f"   Test: {len(cat_test)} Samples für {cat}")
        
        # Verwende optimierte Parameter oder Standard-Parameter
        if optimized_params and cat in optimized_params:
            params = optimized_params[cat]['best_params']
            print(f"🎯 Verwende optimierte Parameter für {cat}:")
            for param, value in params.items():
                print(f"     {param}: {value}")
        else:
            # Standard-Parameter (optimiert für bessere Ergebnisse)
            params = {
                'learning_rate': 2e-5,
                'batch_size': 8,  # Größere Batch-Size für bessere Stabilität
                'epochs': 3,      # Mehr Epochs für bessere Konvergenz
                'max_length': 256,
                'weight_method': 'balanced'
            }
            print(f"📋 Verwende Standard-Parameter für {cat}")
        
        # Erstelle Trainer
        trainer = BERTSentimentTrainer()
        trainer.setup_model()
        
        # Class Weights
        if params.get('weight_method', 'balanced') == 'balanced':
            class_weights = trainer.calculate_class_weights(cat_train['rating'].values)
        else:
            class_weights = None
        
        # Erstelle DataLoaders mit optimierten Parametern
        train_loader, valid_loader = trainer.create_data_loaders(
            cat_train['text'].values,
            cat_train['rating'].values - 1,  # BERT erwartet 0-4 statt 1-5
            cat_valid['text'].values,
            cat_valid['rating'].values - 1,
            batch_size=params.get('batch_size', 4),
            max_length=params.get('max_length', 256)
        )
        
        # Training mit optimierten Parametern
        training_results = trainer.train(
            train_loader, valid_loader, 
            epochs=params.get('epochs', 1), 
            learning_rate=params.get('learning_rate', 2e-5)
        )
        
        # Test Evaluation
        test_dataset = SentimentDataset(
            cat_test['text'].values,
            cat_test['rating'].values - 1,
            trainer.tokenizer,
            max_length=params.get('max_length', 256)
        )
        test_loader = DataLoader(test_dataset, batch_size=params.get('batch_size', 4), shuffle=False)
        
        _, test_acc, test_preds, test_labels = trainer.eval_epoch(test_loader, None)
        
        # Metriken berechnen
        test_labels = [l + 1 for l in test_labels]  # Zurück zu 1-5
        test_preds = [p + 1 for p in test_preds]
        
        classification_rep = classification_report(
            test_labels, test_preds, 
            output_dict=True, 
            target_names=['1', '2', '3', '4', '5']
        )
        
        # Speichere Ergebnisse
        results[cat] = {
            'valid_accuracy': training_results['best_accuracy'],
            'test_accuracy': test_acc.item(),
            'classification_report': classification_rep,
            'training_history': {
                'train_losses': training_results['train_losses'],
                'valid_losses': training_results['valid_losses'],
                'train_accuracies': training_results['train_accuracies'],
                'valid_accuracies': training_results['valid_accuracies']
            },
            'hyperparameters': params
        }
        
        # Speichere Modell
        model_path = f'models/bert_{cat.lower()}.pkl'
        trainer.save_model(model_path.replace('.pkl', ''))
        models[cat] = trainer
        
        print(f"✅ {cat} Training abgeschlossen!")
        print(f"   Valid Accuracy: {training_results['best_accuracy']:.4f}")
        print(f"   Test Accuracy: {test_acc.item():.4f}")
    
    return results, models


def save_bert_results(results, categories):
    """Speichert BERT Training-Ergebnisse als finale Ergebnisse."""
    bert_results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'DistilBERT',
        'vectorizer': 'BERT Tokenizer',
        'categories': categories,
        'overall_accuracy': np.mean([results[cat]['test_accuracy'] for cat in categories]),
        'category_results': results
    }
    
    # Speichern als finale Ergebnisse (überschreibt baseline_results.json)
    results_path = Path('results/results.json')
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(bert_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Finale Ergebnisse gespeichert: {results_path}")
    
    # Zusätzlich als BERT-spezifische Datei speichern
    bert_path = Path('results/bert_results.json')
    with open(bert_path, 'w', encoding='utf-8') as f:
        json.dump(bert_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 BERT-spezifische Ergebnisse gespeichert: {bert_path}")


def main():
    """Hauptfunktion für BERT Training."""
    print("🤖 Starte BERT/DistilBERT Fine-Tuning...")
    
    # Lade Daten
    data = load_data_for_bert()
    if data is None:
        return
    
    train_df, valid_df, test_df = data
    categories = ['Automotive', 'Books', 'Video_Games']
    
    # Organisiere Daten
    data_dict = {
        'train': train_df,
        'valid': valid_df,
        'test': test_df
    }
    
    # Trainiere BERT Modelle
    results, models = train_bert_models(data_dict, categories)
    
    # Speichere Ergebnisse
    save_bert_results(results, categories)
    
    # Berechne Gesamt-Accuracy
    overall_accuracy = np.mean([results[cat]['test_accuracy'] for cat in categories])
    
    print(f"\n🎯 BERT Training abgeschlossen!")
    print(f"📊 Gesamt-Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
    
    # Vergleich mit Baseline
    baseline_path = Path('results/baseline_results.json')
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
        
        baseline_acc = baseline_results['overall_accuracy']
        improvement = overall_accuracy - baseline_acc
        
        print(f"📈 Vergleich mit Baseline:")
        print(f"   Baseline: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
        print(f"   BERT: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
        print(f"   Verbesserung: {improvement:.4f} ({improvement*100:.1f} Prozentpunkte)")


if __name__ == "__main__":
    main() 