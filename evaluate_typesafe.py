
import pandas as pd

df = pd.read_csv("typesafe_human_review.csv")

df = df.dropna(subset=["human_decision"]).copy()
df["human_decision"] = df["human_decision"].str.strip()

df = df[df["human_decision"].isin(["Include", "Exclude"])]

if df.empty:
    raise ValueError("No completed human decisions found.")

df["predicted_decision"] = df["revised_recommendation"].replace(
    {"Exclude Candidate": "Exclude"}
)

print("\nHuman vs. TypeSafe:")
print(
    pd.crosstab(
        df["human_decision"],
        df["predicted_decision"],
        dropna=False,
    )
)

false_exclusions = df[
    (df["human_decision"] == "Include")
    & (df["predicted_decision"] == "Exclude")
]

print("\nPotential false exclusions:", len(false_exclusions))

for title in false_exclusions["Title"]:
    print("-", title)

print("\nTotal human-reviewed articles:", len(df))
