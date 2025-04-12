import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# IMPORT DATA
df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\Spotify+Streaming+History\\spotify_history.csv")



#AVERAGE PLAY TIME
average_play_time = np.mean(df['ms_played'])
print(f" Average Play Time: {average_play_time / 1000:.2f} seconds")

# Top 5 most played tracks
top_tracks = df.groupby('track_name')['ms_played'].sum().sort_values(ascending=False).head()
print("\n Top 5 Most Played Tracks:")
print(top_tracks)

# Missing values
print("\n Missing Values:")
print(df.isnull().sum())

# Fill missing reasons with 'unknown'
df['reason_start'] = df['reason_start'].fillna('unknown')
df['reason_end'] = df['reason_end'].fillna('unknown')

# Drop duplicates
df.drop_duplicates(inplace=True)

# Top 10 Artists by Total Listening Time
top_artists = df.groupby('artist_name')['ms_played'].sum().sort_values(ascending=False).head(10) / 60000

sns.barplot(x=top_artists.values, y=top_artists.index, hue=top_artists.index,palette='magma', dodge=False,legend=False)
plt.title(" Top 10 Artists by Listening Time (Minutes)")
plt.xlabel("Listening Time (min)")
plt.show()

#  Which Artists Get Skipped Most?
skipped_artists = df[df['skipped']].groupby('artist_name').size().sort_values(ascending=False).head(10)

sns.barplot(x=skipped_artists.values, y=skipped_artists.index,hue=skipped_artists.index, palette='Reds')
plt.title(" Most Skipped Artists")
plt.xlabel("Skip Count")
plt.show()
#  Listening Duration When Shuffle is ON vs OFF
sns.boxplot(x='shuffle', y='ms_played', data=df,hue='shuffle', palette='Set2')
plt.yscale('log')
plt.title(" Listening Time vs. Shuffle Mode")
plt.ylabel("Milliseconds Played (log scale)")
plt.xlabel("Shuffle Mode")
plt.show()

#  Skip Rate by Shuffle Mode
shuffle_skip_rate = df.groupby('shuffle')['skipped'].mean()

sns.barplot(x=shuffle_skip_rate.index, y=shuffle_skip_rate.values,hue=shuffle_skip_rate, palette='Set1')
plt.title("Skip Rate vs. Shuffle Mode")
plt.ylabel("Skip Rate")
plt.xticks([0,1], ['Shuffle OFF', 'Shuffle ON'])
plt.show()

#DISTRIBUTION OF TRACK PLAY DURATION

plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

sns.histplot(df['ms_played'], bins=50, kde=True, color='#00FEC6', edgecolor='#00FFAA', linewidth=1.5)
plt.title(' Distribution of Track Play Durations (Log Scale)', fontsize=16, fontweight='bold', color='white')
plt.xlabel('Milliseconds Played (Log Scale)', fontsize=13, color='white')
plt.ylabel('Number of Plays', fontsize=13, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


# Compute correlation matrix
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, linecolor='gray', cbar=True)

plt.title(" Correlation Matrix", fontsize=16, fontweight='bold', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.show()
# Compute covariance matrix
covariance_matrix = df.cov(numeric_only=True)
print(" Covariance Matrix:\n")
print(covariance_matrix.round(2))





