# BA_Projekt_Local_RAG
# 

Dieses Repository enthält die prototypische Implementierung einer **lokalen KI‑Infrastruktur** zur sicheren Nutzung von Large Language Models (LLMs) im Engineering‑Kontext.  
Der Prototyp kombiniert **semantisches Retrieval (RAG)**, **Open‑Source‑Modelle** und **rein lokale Datenverarbeitung**, um technische Fragen auf Basis interner Dokumente zu beantworten.

---

##  **Funktionen**
- Vollständig **On‑Premise** (keine Cloud, volle Datenhoheit)  
- **RAG‑Pipeline**: semantische Suche + generative Antwort  
- Nutzung von **LLaMA 3B (4‑bit)** via Ollama  
- **ChromaDB** als Vektor‑Datenbank  
- **Streamlit Web‑UI** für einfache Bedienung  
- Antworten **inkl. Quellenpassagen**

---

##  **Technologien**
- **Python**, **LangChain**
- **LLaMA 3B**, **mxbai‑embed‑large**
- **ChromaDB**
- **Streamlit**
- GPU empfohlen (z. B. RTX 3050 Ti)

---

##  **Struktur**
```
app.py               # Entry Point
config.py            # Konfiguration
models/              # LLM + Embeddings
pipeline/            # Indexing + RAG
database/            # Vector Store
ui/                  # Streamlit-UI
data/pdfs/           # Dokumente
```

---

##  **Schnellstart**

### 1. Modelle installieren
```bash
ollama pull llama3:3b
ollama pull mxbai-embed-large
```

### 2. Abhängigkeiten
```bash
pip install -r requirements.txt
```

### 3. Index bauen
```bash
python pipeline/indexing.py
```

### 4. Web-App starten
```bash
streamlit run ui/streamlit_app.py
```

---

##  **Limitierungen**
- Kleines Modell → begrenztes Reasoning  
- Kein RBAC, Monitoring oder DMS‑Integration  
- Prototyp, nicht produktionsbereit  

---

## 👤 **Autor**
**Azmi Zaafouri** – TH Köln  
Bachelorarbeit: *„Lokale KI‑Infrastruktur für sensible Engineering‑Daten: Ein Konzept für mittelständische Unternehmen“*
