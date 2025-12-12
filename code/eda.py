# code/eda.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "used_cars_clean.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # Quick summary
    summary_path = PLOTS_DIR / "eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Basic describe:\n")
        f.write(df.describe(include="all").to_string())
    print(f"Wrote: {summary_path}")

    # Price distribution
    plt.figure()
    df["price"].hist(bins=50)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "price_hist.png")
    plt.close()

    # Mileage vs price
    plt.figure()
    df.plot(kind="scatter", x="milage", y="price", alpha=0.3)
    plt.title("Mileage vs Price")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "milage_vs_price.png")
    plt.close()

    # Price by year (boxplot)
    plt.figure()
    df.boxplot(column="price", by="model_year", grid=False)
    plt.suptitle("")
    plt.title("Price by Model Year")
    plt.xlabel("Model Year")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "price_by_year.png")
    plt.close()

    print("EDA plots saved to plots/.")


if __name__ == "__main__":
    main()
