import re
import pandas as pd
import plotly.express as px

def extract_sql(text: str) -> str:
    """Grab the first SELECT …; block."""
    m = re.search(r"(SELECT[\s\S]*?;)|\bSQL:\s*([\s\S]*?)(?=--|$)", text, re.I)
    return (m.group(1) or m.group(2) or "").strip().rstrip(";")

def auto_chart(df: pd.DataFrame):
    if df.empty or len(df.columns) < 2:
        return None

    x, y = df.columns[0], df.columns[1]

    # PIE CHART for "payment_type" + count
    if "payment_type" in df.columns.str.lower():
        fig = px.pie(df, names=x, values=y, title=f"{y} by {x}")
        return fig

    # LINE CHART for time trends
    if pd.api.types.is_datetime64_any_dtype(df[x]) or "month" in x.lower():
        fig = px.line(df, x=x, y=y, title=f"{y} over Time", markers=True)
        return fig

    # DEFAULT: Bar for everything else
    return px.bar(df, x=x, y=y, title=f"{y} by {x}")