import pandas as pd
import matplotlib.pyplot as plt

# Load the international football results directly from the project CSV
df = pd.read_csv("results.csv")

print("Missing values:")
print(df.isnull().sum())

# Remove incomplete match records so the later counts and comparisons use complete data
df = df.dropna()

print("Number of matches:", len(df))

print("Number of tournaments:", df["tournament"].nunique())

# Convert the date column so matches can be filtered reliably by year
df["date"] = pd.to_datetime(df["date"])

# Filter the converted dates to count only matches played in 2018
matches_2018 = df[df["date"].dt.year == 2018]

print("Football Matches in 2018:", len(matches_2018))


# Compare home and away scores to classify each result as a home win, home loss, or draw
home_wins = (df["home_score"] > df["away_score"]).sum()

home_losses = (df["home_score"] < df["away_score"]).sum()

draws = (df["home_score"] == df["away_score"]).sum()


print("Home wins:", home_wins)

print("Home losses:", home_losses)

print("Draws:", draws)


# A pie chart shows the overall proportion of wins, losses, and draws
plt.pie(
    [home_wins, home_losses, draws],
    labels=["Wins", "Losses", "Draws"],
    autopct="%1.1f%%"
)

plt.show()


# Compare how many matches were played at neutral venues versus non-neutral venues
df["neutral"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.show()