# INFO.md – Product Requirements Document (PRD)

## 1  Zweck des Dokuments
Dieses Dokument spezifiziert alle Anforderungen, Ressourcen, Zeitpläne und Risiken für die Umsetzung von **Aufgabe 1: Sentimentanalyse von Produktrezensionen** des IU‐Projektberichts in „DLBAIPNLP01_D – Projekt: NLP“. Es dient als verbindliche Grundlage für Planung, Umsetzung und Bewertung des Projekts.

## 2  Hintergrund
Online‐Rezensionen sind ein zentraler Informationskanal für Konsument:innen und Unternehmen. Die automatisierte Sentimentanalyse ermöglicht es, große Mengen dieser Texte effizient auszuwerten. Aufgabe 1 fordert die Entwicklung eines Mehrklassen‑Klassifikators, der Amazon‑Produktrezensionen anhand von Titel und Beschreibung in Sternebewertungen von 1 bis 5 einordnet.

## 3  Projektziele
* **Primärziel:** Entwicklung eines reproduzierbaren Textklassifikationssystems (ML‑Pipeline) zur Vorhersage der Sternebewertung (1–5) aus Rezensionstexten.
* **Leistungsziele (KPIs):**
  * Gesamtexaktheit ≥ 70 % (und mindestens 20 Prozentpunkte über einfache Mehrheitsklasse).
  * Confusion‑Matrix, Precision, Recall und F1‑Scores je Klasse ausgewiesen.
  * Separate Genauigkeit je Produktkategorie (≥ 3 Kategorien) berichtet.
* **Dokumentationsziele:** Projektbericht gemäß IU‑Prüfungsleitfaden; vollständiger, kommentierter Code in einem privaten GitHub‑Repository.

## 4  Umfang
### In Scope
1. Datenerhebung (Amazon Reviews ’23).
2. Datenexploration und ‑vorverarbeitung (Tokenisierung, Stop‑Words, Lemmatisierung, Kodierung).
3. Modelltraining (Baseline + fortgeschrittenes Modell).
4. Evaluation & Fehleranalyse.
5. Ergebnisdokumentation und Berichtserstellung.

### Out of Scope
* Echtzeit‑Deployment der Lösung.
* UI‑Entwicklung für Endnutzer:innen.

## 5  Datenanforderungen
| Anforderung | Beschreibung |
|-------------|--------------|
| **Quelle**  | Amazon Reviews ’23 (McAuley Lab). |
| **Kategorien** | Mind. drei unterschiedliche Kategorien (z. B. *Automotive*, *Books*, *Video Games*). |
| **Felder** | `review_title`, `review_body`, `star_rating`. |
| **Mindestumfang** | ≥ 10 000 Rezensionen pro Kategorie (falls verfügbar). |
| **Aufteilung** | 80 % Training / 20 % Test (stratifiziert nach Klassen). |
| **Speicherbedarf** | ≈ 10 GB (JSON + verarbeitete Dateien). |

## 6  Funktionale Anforderungen
1. **Datenbeschaffung:** Skript zum Download und Extrahieren der geforderten Kategorien.
2. **Preprocessing‑Pipeline:** Reinigungs‑ und Feature‑Engineering‑Schritte als wiederverwendbares Python‑Modul.
3. **Baseline‑Modell:** Multinomial Naive Bayes mit TF‑IDF‑Merkmalen.
4. **Fortgeschrittenes Modell:** Transformer‑basierte Architektur (BERT oder DistilBERT fine‑tuned) oder klassischer ML‑Ansatz (SVM/LogReg) – Entscheidung nach Experiment.
5. **Evaluation:** Automatisches Reporting von Accuracy, Confusion‑Matrix, Klassenscores und Kategorie‑spezifischen Metriken.
6. **Experiment‑Tracking:** Einsatz von MLflow oder vergleichbarem Tool; Versionierung aller Artefakte.
7. **Reproduzierbarkeit:** `requirements.txt` / `environment.yml`; deterministische Seeds.
8. **GitHub‑Repository:** Klare Ordnerstruktur, Readme, CI‑Checks für PEP8.

## 7  Nicht‑funktionale Anforderungen
* **Leistung:** Trainingslauf ≤ 4 h auf Google Colab GPU oder lokaler RTX‑GPU.
* **Wartbarkeit:** Modulhafte Code‑Struktur, Unit‑Tests für Kernfunktionen (> 70 % Testabdeckung).
* **Dokumentation:** Inline‑Docstrings (NumPy‑Style) + automatisch generierte API‑Docs.
* **Compliance:** Einhaltung der IU‑Vorgaben zu Plagiatsvermeidung, Formatierung und Zitierweise; korrekte Quellenangabe externer Bibliotheken.
* **Datenschutz:** Verzicht auf personenbezogene Daten; Datensatz ist öffentlich verfügbar.

## 8  Deliverables
1. **INFO.md** (dieses PRD).
2. **GitHub‑Repository** mit Quellcode und README.
3. **Trainiertes Modell** (`.pkl` oder Hugging Face Checkpoint).
4. **Evaluationsberichte** (Confusion‑Matrix‑PNG, Metriken‑CSV).
5. **Projektbericht** (7–10 Seiten) gemäß IU‑Standard.
6. **Präsentationsfolien** (optional für interne Vorstellung).

## 9  Zeitplan & Meilensteine
| KW / Datum | Meilenstein |
|------------|------------|
| **KW 30 (22. – 28. 07.)** | Datensatzdownload & Auswahl der Kategorien abgeschlossen |
| **KW 31 (29. 07. – 04. 08.)** | Preprocessing‑Pipeline implementiert |
| **KW 32 (05. – 11. 08.)** | Baseline‑Modell trainiert + evaluiert |
| **KW 33 (12. – 18. 08.)** | Fortgeschrittenes Modell experimentell validiert |
| **KW 34 (19. – 25. 08.)** | Fehleranalyse & Hyperparameter‑Tuning abgeschlossen |
| **KW 35 (26. 08. – 01. 09.)** | Bericht Version 1 und GitHub‑Code Freeze |
| **KW 36 (02. – 08. 09.)** | Finale Review, Formatcheck, Abgabe via Turnitin |

## 10  Ressourcen & Budget
* **Hardware:** Eigener Laptop (16 GB RAM) + kostenloses Google Colab GPU‑Kontingent.
* **Software‑Stacks:** Python ≥ 3.11, Pandas, scikit‑learn, NLTK/spaCy, Transformers, MLflow, Matplotlib/Seaborn für Visualisierung.
* **Personal:** Jan Serebranski

## 11  Risiken & Gegenmaßnahmen
| Risiko | Auswirkung | Gegenmaßnahme |
|--------|------------|---------------|
| **Unausgeglichene Klassen** | Verzerrte Metriken | Strat. Sampling, Class Weights, SMOTE |
| **Rechenlimit (Colab)** | Abbruch Training | Datensatzgröße reduzieren, Batch‑Training |
| **Plagiatsvorwurf** | Nichtbestehen | Eigener Code, Quellenangaben, Turnitin‑Check |
| **Deadline‑Verzug** | Punktabzug | Puffer 1 Woche, wöchentl. Fortschrittskontrolle |

---
*Version 0.1 – 22. Juli 2025*

