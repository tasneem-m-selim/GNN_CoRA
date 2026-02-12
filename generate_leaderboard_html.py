import pandas as pd

# Load leaderboard
df = pd.read_csv("final_leaderboard.csv")

# Sort descending by Accuracy
df = df.sort_values(by="Accuracy", ascending=False)

# Optional: assign medals
def medal(i):
    if i == 0:
        return "🥇"
    elif i == 1:
        return "🥈"
    elif i == 2:
        return "🥉"
    return ""

df["Rank"] = [f"{i+1} {medal(i)}" for i in range(len(df))]

# Generate HTML table
html = df.to_html(index=False, columns=["Rank", "Team", "Accuracy"], escape=False)

# Wrap in basic HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>GNN CoRA Leaderboard</title>
<style>
table {{border-collapse: collapse; width: 50%; margin:auto;}}
th, td {{border: 1px solid #ddd; padding: 8px; text-align: center;}}
th {{background-color: #2c3e50; color: white;}}
tr:nth-child(even) {{background-color: #f2f2f2;}}
.gold {{background-color:#FFD700;}}
.silver {{background-color:#C0C0C0;}}
.bronze {{background-color:#CD7F32; color:white;}}
</style>
</head>
<body>
<h1 style="text-align:center">🏆 GNN CoRA Leaderboard</h1>
{html}
</body>
</html>
"""

with open("final_leaderboard.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("Updated final_leaderboard.html")
