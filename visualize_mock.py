import pandas as pd
import matplotlib.pyplot as plt


def plot_stacked_area(csv_path: str = "airo_dashboard_mock_data.csv") -> None:
    """
    Visualise evolution of disclosed AI risks (year-over-year)
    using a stacked area chart as outlined in README.
    """
    df = pd.read_csv(csv_path)

    # Aggregate mention counts by Year and top-level Risk Category
    pivot = (
        df.groupby(["Year", "Risk_Category"])["Mention_Count"]
        .sum()
        .reset_index()
        .pivot(index="Year", columns="Risk_Category", values="Mention_Count")
        .fillna(0)
    )

    years = pivot.index.values
    categories = pivot.columns.tolist()
    values = [pivot[cat].values for cat in categories]

    plt.figure(figsize=(10, 6))
    plt.stackplot(years, values, labels=categories)
    plt.title("Evolution of Disclosed AI Risks (Year-over-Year)")
    plt.xlabel("Year")
    plt.ylabel("Total Mention Count")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_grouped_bars(csv_path: str = "airo_dashboard_mock_data.csv") -> None:
    """
    Alternative grouped bar chart view: per year, grouped by risk category.
    """
    import numpy as np

    df = pd.read_csv(csv_path)
    pivot = (
        df.groupby(["Year", "Risk_Category"])["Mention_Count"]
        .sum()
        .reset_index()
        .pivot(index="Year", columns="Risk_Category", values="Mention_Count")
        .fillna(0)
    )

    years = pivot.index.values
    categories = pivot.columns.tolist()

    x = np.arange(len(years))
    width = 0.8 / len(categories)  # total width ~0.8, split across categories

    plt.figure(figsize=(10, 6))
    for i, category in enumerate(categories):
        plt.bar(
            x + i * width,
            pivot[category].values,
            width,
            label=category,
        )

    plt.title("Disclosed AI Risks by Year and Category")
    plt.xlabel("Year")
    plt.ylabel("Total Mention Count")
    plt.xticks(x + width * (len(categories) - 1) / 2, years)
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Default: stacked area chart as suggested in README
    plot_stacked_area()



