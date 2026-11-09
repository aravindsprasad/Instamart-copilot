"""Small shared helpers: markdown tables and report-fragment writing (standard library + pandas only)."""
import os
import pandas as pd

REPORTS = "reports"
os.makedirs(REPORTS, exist_ok=True)


def md_table(df, index=False, floatfmt="{:,.2f}"):
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class Report:
    """Collects markdown, prints it and saves it as reports/<name>.md (merged into README by build_readme.py)."""
    def __init__(self, name):
        self.name, self.parts = name, []

    def add(self, text=""):
        print(text)
        self.parts.append(text)

    def save(self):
        with open(os.path.join(REPORTS, self.name + ".md"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(self.parts) + "\n")
