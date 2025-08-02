#!/usr/bin/env python
"""
generate_checksums.py – SHA-Checksums für alle Modelle generieren

Ziel: Erstellung von SHA-256 Checksums für alle trainierten Modelle
zur Sicherstellung der Reproduzierbarkeit und Integrität.
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime

def generate_sha_checksum(filepath):
    """Generiert SHA-256 Checksum für eine Datei."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_file_size(filepath):
    """Gibt die Dateigröße in MB zurück."""
    size_bytes = Path(filepath).stat().st_size
    return round(size_bytes / (1024 * 1024), 2)

def main():
    """Hauptfunktion zur Generierung aller Checksums."""
    
    # Modelle und ihre Pfade
    models = {
        "baseline_automotive": "models/baseline_automotive.pkl",
        "baseline_books": "models/baseline_books.pkl", 
        "baseline_video_games": "models/baseline_video_games.pkl",
        "bert_automotive": "models/bert_automotive/model.safetensors",
        "bert_books": "models/bert_books/model.safetensors",
        "bert_video_games": "models/bert_video_games/model.safetensors"
    }
    
    checksums = {}
    
    print("🔍 Generiere SHA-Checksums für alle Modelle...")
    
    for model_name, model_path in models.items():
        if Path(model_path).exists():
            sha = generate_sha_checksum(model_path)
            size_mb = get_file_size(model_path)
            
            checksums[model_name] = {
                "filepath": model_path,
                "sha256": sha,
                "size_mb": size_mb,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ {model_name}: {sha[:16]}... ({size_mb} MB)")
        else:
            print(f"❌ {model_name}: Datei nicht gefunden")
    
    # Speichere Checksums
    output_path = "results/model_checksums.json"
    with open(output_path, 'w') as f:
        json.dump(checksums, f, indent=2)
    
    print(f"\n✅ Checksums gespeichert in: {output_path}")
    
    # Erstelle auch eine Markdown-Tabelle
    md_content = "# Model Checksums\n\n"
    md_content += f"**Generiert am:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "| Modell | SHA-256 | Größe (MB) |\n"
    md_content += "|--------|---------|------------|\n"
    
    for model_name, data in checksums.items():
        md_content += f"| {model_name} | `{data['sha256']}` | {data['size_mb']} |\n"
    
    with open("results/model_checksums.md", 'w') as f:
        f.write(md_content)
    
    print("✅ Markdown-Tabelle erstellt: results/model_checksums.md")

if __name__ == "__main__":
    main() 