# Modellwahl-Analyse: BERT vs. SVM

**Datum:** 30. Juli 2025  
**Entscheider:** Jan Serebranski  
**Baseline-Ergebnisse:** 71.5% Accuracy, Klassenungleichgewicht identifiziert

---

## 🎯 **1. Entscheidungskriterien**

### **Projektanforderungen:**
- **Accuracy ≥ 70%** ✅ (Baseline bereits erfüllt)
- **≥20% Verbesserung über Mehrheitsklasse** ❌ (Nur 2.4-5.8% erreicht)
- **Rechenzeit ≤ 4h** auf Colab GPU
- **Reproduzierbarkeit** und **Dokumentation**

### **Identifizierte Probleme:**
- **Klassenungleichgewicht:** Rating 5 dominiert (60-70%)
- **Schwache Performance bei Rating 2-4:** Recall < 20%
- **Einfache Baseline:** Naive Bayes kann komplexe Muster nicht erfassen

---

## 🤖 **2. Modell-Optionen**

### **Option A: BERT/DistilBERT Fine-Tuning**
**Vorteile:**
- ✅ **Beste Textverständnis:** Kann Kontext und Nuancen erfassen
- ✅ **State-of-the-art** für Sentimentanalyse
- ✅ **Class Weights** einfach implementierbar
- ✅ **Transfer Learning:** Vorgelerntes Wissen nutzen
- ✅ **Gute Performance** bei unausgeglichenen Klassen

**Nachteile:**
- ❌ **Hohe Rechenzeit:** 2-4h Training
- ❌ **GPU erforderlich:** Colab notwendig
- ❌ **Komplexität:** Mehr Hyperparameter
- ❌ **Overfitting-Risiko** bei kleinen Klassen

### **Option B: SVM mit RBF-Kernel**
**Vorteile:**
- ✅ **Schnell:** 30-60min Training
- ✅ **Einfach:** Weniger Hyperparameter
- ✅ **Class Weights** verfügbar
- ✅ **Robust:** Weniger Overfitting-Risiko
- ✅ **CPU-basiert:** Keine GPU nötig

**Nachteile:**
- ❌ **Eingeschränktes Textverständnis:** TF-IDF Features
- ❌ **Feature Engineering** kritisch
- ❌ **Möglicherweise schlechtere Performance** als BERT

---

## 📊 **3. Vergleichsmatrix**

| Kriterium | Gewichtung | BERT | SVM | Gewinner |
|-----------|------------|------|-----|----------|
| **Accuracy-Verbesserung** | 40% | 9/10 | 6/10 | **BERT** |
| **Rechenzeit** | 20% | 4/10 | 9/10 | **SVM** |
| **Klassenungleichgewicht** | 25% | 9/10 | 7/10 | **BERT** |
| **Implementierungskomplexität** | 15% | 5/10 | 8/10 | **SVM** |

**Gesamtscore:** BERT = 7.4/10, SVM = 7.2/10

---

## 🔍 **4. Literatur-Review**

### **BERT für Sentimentanalyse:**
- **Devlin et al. (2019):** BERT erreicht SOTA bei Sentiment Tasks
- **Sun et al. (2019):** DistilBERT 97% der BERT-Performance bei 60% der Größe
- **Class Imbalance:** BERT mit Class Weights zeigt 15-25% Verbesserung

### **SVM für Sentimentanalyse:**
- **Joachims (1998):** SVM effektiv für Textklassifikation
- **Class Imbalance:** SMOTE + SVM zeigt 10-15% Verbesserung
- **Feature Engineering:** N-Grams + Sentiment Lexicons wichtig

---

## 🚀 **5. Empfehlung**

### **✅ Empfohlen: BERT/DistilBERT Fine-Tuning**

**Begründung:**
1. **Höchstes Verbesserungspotential** für Klassenungleichgewicht
2. **State-of-the-art Performance** für Sentimentanalyse
3. **Class Weights** einfach implementierbar
4. **Rechenzeit akzeptabel** (≤4h auf Colab)
5. **Projektziele** erfordern deutliche Verbesserung

### **Implementierungsplan:**
1. **DistilBERT** (kleiner, schneller als BERT)
2. **Class Weights** für ausgewogene Gewichtung
3. **Learning Rate:** 2e-5 bis 5e-5
4. **Batch Size:** 16-32 (GPU-Memory)
5. **Epochs:** 3-5 (Early Stopping)

---

## 📋 **6. Fallback-Plan**

### **Wenn BERT zu langsam/komplex:**
1. **SVM mit RBF-Kernel** als Alternative
2. **Feature Engineering:** N-Grams, Sentiment Lexicons
3. **SMOTE** für Klassenausgleich
4. **Grid Search** für Hyperparameter

### **Wenn BERT nicht konvergiert:**
1. **Kleinere Datensätze** testen
2. **Learning Rate** reduzieren
3. **Batch Size** anpassen
4. **Fallback auf SVM**

---

## ✅ **7. Beschluss**

**Entscheidung:** **BERT/DistilBERT Fine-Tuning**

**Nächste Schritte:**
1. **Aufgabe 4.2:** Fine-Tuning Setup (`train_hf.py`)
2. **Class Weights** implementieren
3. **Hyperparameter-Tuning** mit Optuna
4. **Frühe Evaluation** nach 1-2 Epochs

**Zeitrahmen:** 1-2 Wochen für Training und Evaluation

---

**Datum:** 30. Juli 2025 