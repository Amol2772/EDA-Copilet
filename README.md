# 📊 EDA Copilot

> **An AI agent that performs exploratory data analysis through natural language — no code required.**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://eda-copilet-e37wzufmfj54htixxwxvzu.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-1C3C3C?style=for-the-badge)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)

---

## 🌐 Live Demo

**[eda-copilet-e37wzufmfj54htixxwxvzu.streamlit.app](https://eda-copilet-e37wzufmfj54htixxwxvzu.streamlit.app/)**

---

## 🧩 Problem Statement

Exploratory Data Analysis is the first — and most critical — step in any data science workflow. Yet it remains **time-consuming, repetitive, and inaccessible** to non-technical stakeholders.

**Current pain points:**
- Analysts spend hours writing boilerplate pandas/SQL code just to answer basic questions
- Static profiling reports dump raw numbers with no interpretation or context
- Non-technical users can't query datasets without coding knowledge
- Insights are buried in dataframes — not surfaced in plain English

---

## ✅ Solution

**EDA Copilot** is a conversational AI agent that takes a raw dataset and answers natural language questions about it — by writing and executing real code under the hood.

| Ask | Get |
|---|---|
| *"Which vehicle type has the highest average daily rate?"* | Real answer computed from data, not hallucinated |
| *"Plot the distribution of daily rates"* | Actual chart rendered inline |
| *"Is there a correlation between trips taken and rating?"* | Pearson r, p-value + plain English interpretation |
| *"Using SQL, how many vehicles per fuel type?"* | SQL executed against live data, tabular result |

---

## 🏗️ Architecture

```
User Query (natural language)
        ↓
  Streamlit Chat UI  ←→  Conversation Memory
        ↓
  LangGraph ReAct Agent
        ↓
  ┌──────────────────────────────────────────┐
  │              Tool Selection              │
  ├─────────────────┬────────────────────────┤
  │  python_tool    │  PythonAstREPLTool     │ → pandas computation
  │  plot_column    │  matplotlib            │ → hist / bar / box
  │  correlation    │  scipy.stats.pearsonr  │ → r value + p-value
  │  detect_outlier │  IQR method            │ → bounds + count
  │  sql_query      │  SQLite in-memory      │ → text-to-SQL
  └─────────────────┴────────────────────────┘
        ↓
  LLM: Gemini 2.5 Flash (cloud) / Qwen2.5:14b (local)
        ↓
  Plain English Response + Chart PNG
```

---

## 🔧 Tools

| Tool | Trigger | What it does |
|---|---|---|
| `python_tool` | General questions | Executes pandas code on the `df` dataframe |
| `plot_column` | "plot / chart / show" | Generates hist/bar/box, saves to `reports/` |
| `correlation_test` | "correlation between X and Y" | Pearson r + p-value, numeric cols only |
| `detect_outliers` | "outliers in X" | IQR bounds, count, sample values |
| `sql_query` | "using SQL..." | Runs SQL against in-memory `cars` table |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangChain + LangGraph (`create_react_agent`) |
| LLM (cloud) | Google Gemini 2.5 Flash (free tier) |
| LLM (local) | Ollama + Qwen2.5:14b (GPU-accelerated, unlimited) |
| Code execution | `PythonAstREPLTool` (sandboxed pandas REPL) |
| Plotting | matplotlib (Agg backend) |
| Stats | scipy.stats (Pearson correlation) |
| SQL | SQLite in-memory + `pd.read_sql_query` |
| Baseline profiling | ydata-profiling |
| UI | Streamlit |
| Dataset | Cornell Car Rental Dataset (5,851 rows, 16 cols) |

---

## 📦 Setup

### 1. Clone & Install

```bash
git clone https://github.com/Amol2772/EDA-Copilet.git
cd EDA-Copilet
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. API Key (free, no card required)

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Create API key → copy it
3. Create `.env` in project root:

```
GOOGLE_API_KEY=your_key_here
```

### 3. LLM Toggle

In `agent_core.py`:

```python
USE_OLLAMA = False  # False = Gemini (20 req/day), True = Ollama (local GPU, unlimited)
```

For Ollama:
```bash
ollama pull qwen2.5:14b
```

### 4. Run

```bash
streamlit run app.py
```

---

## 💬 Example Queries

```
Which vehicle type has the highest average daily rate?
Plot rate.daily as a histogram
Is there a correlation between renterTripsTaken and rating?
Detect outliers in rate.daily
Using SQL, how many vehicles are there per fuel type?
What are the top 3 vehicle makes by number of listings?
```

---

## 📊 Dataset

**Cornell Car Rental Dataset** — 5,851 peer-to-peer car rental listings

| Column | Type | Notes |
|---|---|---|
| `vehicle.type` | categorical | car / suv / van / truck / minivan |
| `vehicle.make` | categorical | Tesla, Toyota, BMW... |
| `rate.daily` | float | $20–$1,500 daily rate |
| `rating` | float | 501 missing (8.6%) |
| `renterTripsTaken` | float | 431 zeros (7.4%) |
| `fuelType` | categorical | ELECTRIC / GASOLINE, 75 missing |
| `airportcity` | categorical | Nearest airport |

**Data issues flagged by agent + profiler:**
- `location.country` — constant (all US), drop it
- `rating` — 8.6% missing → impute before modeling
- `rate.daily` — 334 outliers, max $1,500 (luxury vehicles)
- `renterTripsTaken` — 7.4% zeros (new/inactive owners)

---

## 🧪 Evaluation

```bash
python eval/eval_harness.py
```

| # | Test | Result |
|---|---|---|
| 1 | Dataset shape (5851 rows, 16 cols) | ✅ PASS |
| 2 | Top vehicle type by daily rate | ✅ PASS |
| 3 | Missing values in rating (501) | ✅ PASS |
| 4 | Mean rate.daily (~$93) | ✅ PASS |
| 5 | Plotting tool triggered | ✅ PASS |
| 6 | Correlation tool triggered | ✅ PASS |
| 7 | Outlier detection (334, bounds 207.50) | ✅ PASS |
| 8 | SQL tool triggered | ✅ PASS |
| 9 | Top vehicle makes (Tesla/Toyota/BMW) | ✅ PASS |
| 10 | Max daily rate ($1500) | ✅ PASS |

**10/10 — 100% pass rate**

---

## 🗓️ Build Log

| Week | Milestone |
|---|---|
| Week 1 | Environment setup, Gemini API + tool-calling verified |
| Week 2 | Agent + 4 tools built on Titanic → swapped to Car Rental dataset |
| Week 3 | ydata-profiling baseline, LLM insight ranking vs profiler alerts |
| Week 4 | Streamlit chat UI, conversational memory, chart rendering |
| Week 5 | SQL tool, eval harness (10/10), Streamlit Cloud deployment |

---

## 🗂️ Project Structure

```
eda-copilot/
├── agent_core.py          # Agent: LLM toggle, tools, prompt
├── app.py                 # Streamlit chat UI + sidebar
├── baseline_profile.py    # ydata-profiling baseline generator
├── CarRentalDataV1.csv    # Dataset
├── eval/
│   └── eval_harness.py    # 10-query automated eval
├── reports/               # Generated charts (gitignored)
├── requirements.txt
├── runtime.txt            # Python 3.11 pin for Streamlit Cloud
└── README.md
```

---

## 🔄 LLM Comparison

| | Gemini 2.5 Flash | Ollama Qwen2.5:14b |
|---|---|---|
| Cost | Free (20 req/day) | Free (local GPU) |
| Speed | ~15s/query | ~20s/query |
| Tool-calling | Excellent | Good |
| Privacy | Sent to Google | Fully local |
| Best for | Final demo | Development |

---

## 📄 License

MIT
