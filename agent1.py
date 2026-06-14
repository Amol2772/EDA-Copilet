import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()
os.makedirs("reports", exist_ok=True)

df = pd.read_csv("CarRentalDataV1.csv")
col_map = {c.lower(): c for c in df.columns}

python_tool = PythonAstREPLTool(locals={"df": df})

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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

agent = create_react_agent(
    llm,
    tools=[python_tool, plot_column, correlation_test, detect_outliers],
    prompt=f"You are a data analyst. A pandas dataframe `df` (car rental dataset) is available. Columns: {list(df.columns)}. Use python tool for general computation, plot_column for charts, correlation_test for correlations between numeric columns, detect_outliers for outlier detection. Don't guess numbers — always use tools."
)

def extract_text(result):
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(b["text"] for b in content if b.get("type") == "text")
    return content
alerts_text = """
- location.country: constant value (no variation)
- fuelType: 75 (1.3%) missing values
- rating: 501 (8.6%) missing values
- renterTripsTaken: 431 (7.4%) zeros
- reviewCount: 501 (8.6%) zeros
"""

query = f"""An automated profiling tool flagged these issues in the car rental dataset:
{alerts_text}
For each, briefly explain why it matters for analysis, rank them by importance (most to least actionable), and suggest one concrete follow-up step. Verify at least one claim using the python tool."""

result = agent.invoke({"messages": [{"role": "user", "content": query}]})
for m in result["messages"]:
    print(type(m).__name__, "->", repr(m.content)[:300])
print("---FINAL---")
print(extract_text(result))