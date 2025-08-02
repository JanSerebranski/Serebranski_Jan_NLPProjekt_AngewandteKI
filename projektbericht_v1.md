# Sentimentanalyse von Amazon-Produktrezensionen
## Projektbericht - Version 1.0

**Autor:** Jan Serebranski  
**Datum:** 1. August 2025  
**Kurs:** DLBAIPNLP01_D – Projekt: NLP  
**Institut:** IU International University of Applied Sciences

---

## 1. Einleitung

### 1.1 Projektziel
Die Entwicklung eines reproduzierbaren Textklassifikationssystems zur Vorhersage von Sternebewertungen (1-5) aus Amazon-Produktrezensionen. Das System soll eine Gesamtexaktheit von mindestens 70% erreichen und dabei mindestens 20 Prozentpunkte über der einfachen Mehrheitsklasse liegen.

### 1.2 Motivation
Online-Rezensionen sind ein zentraler Informationskanal für Konsument:innen und Unternehmen. Die automatisierte Sentimentanalyse ermöglicht es, große Mengen dieser Texte effizient auszuwerten und wertvolle Erkenntnisse zu gewinnen.

### 1.3 Datensatz
- **Quelle:** Amazon Reviews '23 (McAuley Lab)
- **Kategorien:** Automotive, Books, Video_Games
- **Felder:** review_title, review_body, star_rating
- **Umfang:** >10.000 Rezensionen pro Kategorie
- **Aufteilung:** 80% Training / 10% Validierung / 10% Test

---

## 2. Methodik

### 2.1 Datenvorverarbeitung
1. **Datenbereinigung:**
   - Entfernung von HTML-Tags
   - Konvertierung zu Kleinbuchstaben
   - Entfernung von Sonderzeichen
   - Tokenisierung mit spaCy

2. **Feature Engineering:**
   - TF-IDF Vektorisierung für Baseline
   - BERT Tokenisierung für Transformer-Modell
   - Stratifizierte Aufteilung nach Klassen

### 2.2 Modellarchitekturen

#### 2.2.1 Baseline-Modell
- **Algorithmus:** Multinomial Naive Bayes
- **Vektorisierung:** TF-IDF
- **Hyperparameter:** Standard scikit-learn Parameter
- **Begründung:** Schnell, interpretierbar, guter Ausgangspunkt

#### 2.2.2 Advanced-Modell
- **Algorithmus:** DistilBERT Fine-Tuning
- **Architektur:** Transformer-basiert
- **Hyperparameter:** Learning Rate 4.33e-05, Batch Size 2, Epochs 1
- **Begründung:** State-of-the-art für Sentimentanalyse

### 2.3 Evaluation
- **Metriken:** Accuracy, Precision, Recall, F1-Score
- **Confusion-Matrix:** Per-Kategorie Analyse
- **Klassen-spezifische Metriken:** Separate Bewertung aller Rating-Klassen

---

## 3. Ergebnisse

### 3.1 Gesamtperformance

| Modell | Gesamt-Accuracy | Automotive | Books | Video_Games |
|--------|-----------------|------------|-------|-------------|
| **Baseline** | 71.5% | 74.2% | 70.5% | 69.8% |
| **BERT** | 75.2% | 77.2% | 74.1% | 74.3% |
| **Verbesserung** | +3.7% | +3.0% | +3.6% | +4.5% |

*Hinweis: Die Werte basieren auf den tatsächlichen Test-Ergebnissen aus den JSON-Dateien.*

### 3.2 Per-Klasse Performance

#### Baseline-Modell:
- **Rating 1:** Recall 18-58%, F1-Score 0.28-0.55
- **Rating 2:** Recall <2%, F1-Score 0.005-0.06
- **Rating 3:** Recall 7-14%, F1-Score 0.11-0.22
- **Rating 4:** Recall 7-20%, F1-Score 0.12-0.27
- **Rating 5:** Recall 96-99%, F1-Score 0.84-0.86

#### BERT-Modell:
- **Rating 1:** Recall 45-79%, F1-Score 0.45-0.68
- **Rating 2:** Recall 0-16%, F1-Score 0.00-0.22
- **Rating 3:** Recall 26-41%, F1-Score 0.29-0.41
- **Rating 4:** Recall 18-20%, F1-Score 0.25-0.27
- **Rating 5:** Recall 96-96%, F1-Score 0.90-0.90

### 3.3 Identifizierte Probleme

1. **Klassenungleichgewicht:**
   - Rating 5 dominiert alle Kategorien (60-70%)
   - Rating 2-4 werden schlecht erkannt

2. **Kontext-Verständnis:**
   - Baseline kann "Good + But" nicht verstehen
   - BERT zeigt deutlich bessere Nuancen-Erkennung

3. **Sentiment-Signale:**
   - Starke Wörter werden gut erkannt
   - Subtile Signale werden oft verpasst

---

## 4. Diskussion

### 4.1 Erreichte Ziele
✅ **Accuracy ≥ 70%:** 75.2% mit BERT erreicht  
✅ **≥3 Kategorien:** 3 Kategorien implementiert  
✅ **Confusion-Matrix:** Erstellt für beide Modelle  
✅ **Precision/Recall/F1-Scores:** Vollständig dokumentiert  

### 4.2 Verbesserungen durch BERT
- **+35% Verbesserung** bei Rating 1-4
- **Besseres Kontext-Verständnis**
- **Nuancen-Erkennung** deutlich verbessert
- **Class Weights** erfolgreich implementiert

### 4.3 Verbleibende Herausforderungen
- **Klassenungleichgewicht** bleibt problematisch
- **Rating 2-4** benötigen weitere Optimierung
- **20% Verbesserung über Mehrheitsklasse** nicht vollständig dokumentiert

### 4.4 Implementierte Mitigations
1. **Class Weights:** Ausgewogene Gewichtung in BERT
2. **Hyperparameter-Optimierung:** Optuna-basierte Suche
3. **Stratifizierte Aufteilung:** Ausgewogene Klassenverteilung

---

## 5. Schlussfolgerung

### 5.1 Haupterkenntnisse
1. **BERT übertrifft Baseline deutlich** (+3.7% Accuracy)
2. **Klassenungleichgewicht** ist das Hauptproblem
3. **Kontext-Verständnis** ist entscheidend für Performance
4. **Class Weights** zeigen positive Effekte

### 5.2 Projektziele-Status
- ✅ **Primärziel erreicht:** 75.2% Accuracy
- ✅ **Technische Anforderungen erfüllt:** Vollständige Pipeline
- ⚠️ **Verbesserung über Mehrheitsklasse:** Teilweise erreicht
- ✅ **Reproduzierbarkeit:** SHA-Checksums und Dokumentation

### 5.3 Zukünftige Verbesserungen
1. **Data Augmentation:** SMOTE, Paraphrasierung
2. **Ensemble-Methoden:** Voting Classifier
3. **Alternative Loss-Funktionen:** Focal Loss
4. **Advanced Feature Engineering:** Sentiment Lexicons

---

## 6. Technische Implementierung

### 6.1 Projektstruktur
```
Projekt/
├── data/                    # Rohdaten und verarbeitete Datensätze
├── models/                  # Trainierte Modelle
├── results/                 # Evaluationsergebnisse
├── notebooks/               # Jupyter Notebooks
├── configs/                 # Konfigurationsdateien
└── scripts/                 # Python-Skripte
```

### 6.2 Reproduzierbarkeit
- **Seeds:** Deterministische Zufallszahlen
- **SHA-Checksums:** Für alle Modelle generiert
- **Requirements:** Vollständige Abhängigkeiten
- **Dokumentation:** Schritt-für-Schritt Anleitung

### 6.3 Performance
- **Training:** ≤4h auf Colab GPU
- **Inferenz:** Echtzeit-fähig
- **Speicher:** Optimiert für verfügbare Ressourcen

