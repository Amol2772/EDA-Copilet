import os
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from dotenv import load_dotenv
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()
os.makedirs("reports", exist_ok=True)

df = pd.read_csv("CarRentalDataV1.csv")
col_map = {c.lower(): c for c in df.columns}

python_tool = PythonAstREPLTool(locals={"df": df})

conn = sqlite3.connect(":memory:", check_same_thread=False)
df.to_sql("cars", conn, index=False, if_exists="replace")

@tool
def sql_query(query: str) -> str:
    """Run a SQL query against the 'cars' table. Quote column names with dots in double quotes e.g. "rate.daily"."""
    try:
        result = pd.read_sql_query(query, conn)
        return result.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {type(e).__name__}: {e}"

@tool
def plot_column(column: str, kind: str = "hist") -> str:
    """Plot a column from the dataframe. kind: 'hist', 'bar', or 'box'. Returns saved PNG path."""
    column = col_map.get(column.lower())
    if column is None:
        return f"Error: column not found. Available: {list(df.columns)}"
    fig, ax = plt.subplots()
    if kind == "hist":
        df[column].hist(ax=ax)
    elif kind == "bar":
        df[column].value_counts().plot(kind="bar", ax=ax)
    elif kind == "box":
        df.boxplot(column=column, ax=ax)
    ax.set_title(f"{column} ({kind})")
    safe_name = column.replace(".", "_")
    path = f"reports/{safe_name}_{kind}.png"
    fig.savefig(path)
    plt.close(fig)
    return path

@tool
def correlation_test(col1: str, col2: str) -> str:
    """Compute Pearson correlation and p-value between two numeric columns."""
    c1, c2 = col_map.get(col1.lower()), col_map.get(col2.lower())
    if c1 is None or c2 is None:
        return f"Error: column not found. Available: {list(df.columns)}"
    sub = df[[c1, c2]].dropna()
    if not pd.api.types.is_numeric_dtype(sub[c1]) or not pd.api.types.is_numeric_dtype(sub[c2]):
        return f"Error: both columns must be numeric. {c1} and {c2} dtypes: {df[c1].dtype}, {df[c2].dtype}"
    r, p = scipy_stats.pearsonr(sub[c1], sub[c2])
    return f"Pearson correlation between {c1} and {c2}: r={r:.3f}, p-value={p:.4f}"

@tool
def detect_outliers(column: str) -> str:
    """Detect outliers in a numeric column using IQR method. Returns count, bounds, and example values."""
    col = col_map.get(column.lower())
    if col is None:
        return f"Error: column not found. Available: {list(df.columns)}"
    series = df[col].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return f"Error: {col} is not numeric (dtype: {series.dtype})"
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return f"{col}: {len(outliers)} outliers (IQR bounds: {lower:.2f} to {upper:.2f}). Sample values: {outliers.tolist()[:10]}"

# ---- LLM selection ----
USE_OLLAMA = False  # True = local Ollama (no limits), False = Gemini 2.5 Flash (20/day free tier)

if USE_OLLAMA:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen2.5:14b", temperature=0)
else:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

agent = create_react_agent(
    llm,
    tools=[python_tool, plot_column, correlation_test, detect_outliers, sql_query],
    prompt=f"You are a data analyst. A pandas dataframe `df` (car rental dataset) is available. Columns: {list(df.columns)}. Use python tool for general computation, plot_column for charts, correlation_test for correlations between numeric columns, detect_outliers for outlier detection, and sql_query for SQL queries against the 'cars' table (useful for grouping/aggregation). Column names with dots (e.g. rate.daily) must be quoted with double quotes in SQL. Don't guess numbers — always use tools."
)

def extract_text(result):
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(b["text"] for b in content if b.get("type") == "text")
    return content