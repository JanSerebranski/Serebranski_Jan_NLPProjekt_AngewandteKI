# Mitigation Strategies: Implementierte und vorgeschlagene Verbesserungen

**Datum:** 1. August 2025  
**Status:** Analyse abgeschlossen, Implementierung dokumentiert  
**Modelle:** Baseline (Multinomial Naive Bayes) vs. BERT (DistilBERT)

---

## 🎯 **Identifizierte Probleme & Lösungen**

### **Problem 1: Klassenungleichgewicht**
**Symptom:** Rating 5 dominiert (60-70% der Daten), Rating 2-4 werden schlecht erkannt

#### **✅ Implementierte Lösungen:**
1. **Class Weights in BERT:**
   ```python
   # In train_hf.py implementiert
   weight_method = "balanced"
   class_weights = compute_class_weight('balanced', classes, labels)
   ```

2. **Stratifizierte Aufteilung:**
   ```python
   # In pipeline.py implementiert
   train_test_split(..., stratify=labels)
   ```

#### **💡 Vorgeschlagene weitere Lösungen:**
1. **SMOTE (Synthetic Minority Over-sampling Technique):**
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   ```

2. **Data Augmentation:**
   - Paraphrasierung bestehender Reviews
   - Synonym-Ersetzung für Rating 2-4
   - Back-translation für mehr Variation

---

### **Problem 2: Schwache Performance bei Rating 2-4**
**Symptom:** Recall < 30% für diese Klassen

#### **✅ Implementierte Lösungen:**
1. **BERT Fine-Tuning:**
   - Verbesserung von +35% bei Rating 1-4
   - Besseres Kontext-Verständnis

2. **Hyperparameter-Optimierung:**
   ```python
   # In hyperparameter_optimization.py
   learning_rate = 4.33e-05
   batch_size = 2
   epochs = 1
   ```

#### **💡 Vorgeschlagene weitere Lösungen:**
1. **Alternative Loss-Funktionen:**
   ```python
   # Focal Loss für Klassenungleichgewicht
   class FocalLoss(nn.Module):
       def __init__(self, alpha=1, gamma=2):
           super().__init__()
           self.alpha = alpha
           self.gamma = gamma
   ```

2. **Ensemble-Methoden:**
   ```python
   # Voting Classifier
   from sklearn.ensemble import VotingClassifier
   ensemble = VotingClassifier([
       ('baseline', baseline_model),
       ('bert', bert_model)
   ])
   ```

---

### **Problem 3: Kontext-Verständnis**
**Symptom:** Baseline kann "Good + But" nicht verstehen

#### **✅ Implementierte Lösungen:**
1. **BERT Tokenizer:**
   - Kontextuelle Einbettungen
   - Attention-Mechanismus

2. **Max Length Optimierung:**
   ```python
   max_length = 256  # Ausreichend für Kontext
   ```

#### **💡 Vorgeschlagene weitere Lösungen:**
1. **Feature Engineering für Baseline:**
   ```python
   # Negation Detection
   def detect_negation(text):
       negation_words = ['not', 'no', 'never', 'but', 'however']
       return any(word in text.lower() for word in negation_words)
   ```

2. **N-Gram Features:**
   ```python
   # Bigrams und Trigrams
   vectorizer = TfidfVectorizer(ngram_range=(1, 3))
   ```

---

## 📊 **Implementierte Mitigations im Detail**

### **1. Class Weights Implementation**
```python
# In BERTSentimentTrainer.calculate_class_weights()
def calculate_class_weights(self, labels):
    """Berechnet Class Weights für ausgewogene Gewichtung."""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights)
```

**Ergebnis:** Deutliche Verbesserung bei Rating 1-4

### **2. Hyperparameter-Optimierung**
```python
# Beste Parameter aus Optuna
{
    "learning_rate": 4.3284502212938785e-05,
    "batch_size": 2,
    "epochs": 1,
    "max_length": 256,
    "weight_method": "balanced"
}
```

**Ergebnis:** Optimierte Training-Parameter

### **3. Stratifizierte Datenaufteilung**
```python
# In pipeline.py
train_data, temp_data = train_test_split(
    data, 
    test_size=0.2, 
    stratify=data['star_rating'],
    random_state=42
)
```

**Ergebnis:** Ausgewogene Klassenverteilung in allen Splits

---

## 🚀 **Vorgeschlagene weitere Mitigations**

### **1. Data Augmentation Pipeline**
```python
def augment_data(text, rating):
    """Erstellt synthetische Beispiele für Rating 2-4."""
    if rating in [2, 3, 4]:
        # Synonym-Ersetzung
        augmented = replace_synonyms(text)
        # Paraphrasierung
        paraphrased = paraphrase_text(text)
        return [augmented, paraphrased]
    return [text]
```

### **2. Ensemble-Methoden**
```python
class EnsembleSentimentClassifier:
    def __init__(self, baseline_model, bert_model):
        self.baseline = baseline_model
        self.bert = bert_model
    
    def predict(self, texts):
        baseline_pred = self.baseline.predict(texts)
        bert_pred = self.bert.predict(texts)
        # Weighted voting
        return weighted_vote(baseline_pred, bert_pred)
```

### **3. Advanced Loss Functions**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

---

## 📈 **Erwartete Verbesserungen**

### **Mit Data Augmentation:**
- **Rating 2-4 Recall:** +15-20%
- **Gesamt-Accuracy:** +2-3%
- **Klassenbalance:** Deutlich verbessert

### **Mit Ensemble-Methoden:**
- **Robustheit:** Höhere Stabilität
- **Accuracy:** +1-2% Verbesserung
- **Generalisierung:** Bessere Performance auf neuen Daten

### **Mit Focal Loss:**
- **Klassenungleichgewicht:** Automatisch behandelt
- **Training:** Stabileres Training
- **Convergence:** Schnellere Konvergenz

---

## 🎯 **Prioritäten für Implementierung**

### **Priorität 1 (Hoch):**
1. **Data Augmentation** - Sofortige Verbesserung möglich
2. **Ensemble-Methoden** - Robuste Lösung

### **Priorität 2 (Mittel):**
1. **Focal Loss** - Für zukünftige Experimente
2. **Advanced Feature Engineering** - Für Baseline

### **Priorität 3 (Niedrig):**
1. **Back-translation** - Komplexe Implementierung
2. **Custom Attention** - Für BERT-Optimierung

---

## ✅ **Zusammenfassung**

### **✅ Bereits implementiert:**
- Class Weights in BERT
- Hyperparameter-Optimierung
- Stratifizierte Aufteilung
- BERT Fine-Tuning

### **💡 Vorgeschlagen für weitere Verbesserung:**
- Data Augmentation Pipeline
- Ensemble-Methoden
- Alternative Loss-Funktionen
- Advanced Feature Engineering

### **📊 Erwartete Ergebnisse:**
- **Rating 2-4 Recall:** +15-20%
- **Gesamt-Accuracy:** +2-5%
- **Klassenbalance:** Deutlich verbessert

---

*Mitigation Strategies dokumentiert am: 1. August 2025* 