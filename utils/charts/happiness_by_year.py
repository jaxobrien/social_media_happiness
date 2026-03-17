import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_happiness_by_year(df):
    """Line chart: mean happiness index by year, one line per age (10–16)."""

    ages = [10, 11, 12, 13, 14, 15, 16]

    grouped = (
        df[df["age"].isin(ages)]
        .groupby(["year", "age"], observed=True)["happiness_index"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("crest", len(ages))

    for i, age in enumerate(ages):
        sub = grouped[grouped["age"] == age]
        ax.plot(sub["year"], sub["happiness_index"], marker="o",
                label=f"Age {int(age)}", color=palette[i], linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Happiness Index")
    ax.set_title("Average Happiness Over Time by Age")
    ax.legend(title="Age")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig