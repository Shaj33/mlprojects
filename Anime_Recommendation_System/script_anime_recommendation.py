# https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt
from sklearn import cluster
import time

#Association Analysis
print("Association Analysis")

ratings = pd.read_csv("rating.csv")
anime = pd.read_csv("anime.csv")

# Data Preprocessing. Finding all reviews that positively score, and then group them by user to create rulesets

# Drop all non-scored reviews
ratings.drop(ratings[ratings.rating == -1].index, inplace=True)

# Drop all non-TV anime
anime.drop(anime[anime.type != 'TV'].index, inplace=True)

anime = anime.sort_values(by="members", ascending=False).head(1000)

# Set anime id to names, for better reading
ratings['anime_id'] = ratings["anime_id"].map(anime.set_index('anime_id')['name'])

# Drop any ratings below 6, which would signify the show is more disliked than liked
ratings.drop(ratings[ratings.rating < ratings.rating.mean() ].index, inplace=True)

# Remove any na values
ratings.dropna(inplace=True)

ratings = ratings.groupby("user_id")['anime_id'].apply(list).reset_index(name='List')

ratings['# of Reviews'] = ratings["List"].str.len()

print(ratings.head(10))

results = apriori(ratings['List'], min_confidence=0.5, min_support=0.05)

StartTime = time.time()
rules = pd.DataFrame(columns=("Set", "Recommend", "Support", "Confidence"))

for itemset in results:
    for rule_index in range(len(itemset.ordered_statistics)):
        rules = rules._append({"Set": list(itemset.ordered_statistics[rule_index].items_base), 
                               "SetSize": len(list(itemset.ordered_statistics[rule_index].items_base)),
                              "Recommend": list(itemset.ordered_statistics[rule_index].items_add), 
                              'Support': itemset.support, 
                              'Confidence': itemset.ordered_statistics[rule_index].confidence}, ignore_index=True)
endTime = time.time()
rules.to_csv("rules.csv", index=False)

print("Done")

print("Time:", endTime - StartTime)

rules = pd.read_csv("rules.csv")

def query_anime(name, rules):
    temp = rules.copy()
    print(temp.shape)
    temp['Set']=temp['Set'].apply(lambda x:x if name in x else 'diff')
    temp=temp[temp['Set']!='diff']
    temp=temp.reset_index(drop=True)
    print(temp.sort_values(by=['SetSize', 'Confidence'], ascending=[True, False]).head(10).to_string())
    print(temp.shape)

name = "Kuroko no Basket"

print(anime.sort_values(by='members', ascending=False).head(10))

query_anime(name, rules)

print(rules.sort_values(by=['SetSize', 'Confidence'], ascending=[True, False]).head(20).to_string())


# Clustering Analysis
print("Clustering Analysis")

ratings = pd.read_csv('rating.csv')
anime = pd.read_csv('anime.csv')

# Data Preprocessing. Finding all reviews that positively score, and then group them by user to create a vector for K-Means clustering

# Drop all non-scored reviews
ratings.drop(ratings[ratings.rating == -1].index, inplace=True)

# Drop all non-TV anime
anime.drop(anime[anime.type != 'TV'].index, inplace=True)

# Assign Dummy variables according to genre for each anime
genres = anime['genre'].str.split(',', expand=True).apply(lambda x: [e.strip() if type(e) == str else 'None' for e in x ])
genres = pd.get_dummies(genres.stack(), dtype=float).groupby(level=0).max()
genres.drop("None", axis=1, inplace=True)
genre_labels = list(genres.columns)
anime.drop(['genre'], inplace=True, axis=1)
anime = pd.concat([anime, genres], axis=1)

# Assign those dummy variables to each review
for column in genre_labels:
    ratings[column] = ratings["anime_id"].map(anime.set_index('anime_id')[column])

ratings['# of Reviews'] = 1

# Drop any ratings below 6, which would signify the show is more disliked than liked
ratings.drop(ratings[ratings.rating < ratings.rating.mean()].index, inplace=True)

# Remove any na values
ratings.dropna(inplace=True)

# Sum User Reviews
ratings = ratings.groupby("user_id")[genre_labels + ['# of Reviews']].sum()

# Remove few review users, which might create outlier vectors
ratings.drop(ratings[ratings['# of Reviews'] < 5].index, inplace=True)

for column in genre_labels:
    ratings[column] = ratings[column]/ratings['# of Reviews']
print(ratings)

# Dropping columns that have a low mean and standard distribution (Little to no viewers, and little to no die-hard viewers)
for column in genre_labels:
    if ratings[column].mean() < 0.1 and ratings[column].std() < 0.1:
        print(column, "dropped")
        ratings.drop(column, inplace=True, axis=1)
        continue

plt.show()
ratings.drop('Ecchi', inplace=True, axis=1)
genre_labels = ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Harem', 'Magic', 'Mystery', 'Psychological', 'Romance', 'School', 'Sci-Fi', 'Seinen', 'Shoujo', 'Shounen', 'Slice of Life', 'Super Power','Supernatural']

pd.plotting.scatter_matrix(ratings[genre_labels].head(500), diagonal='hist')
plt.show()

#K-Means Clustering
sse_list=[]

#Plotting the optimal number of clusters
for cluster_count in range(1,20):
    kMeans = cluster.KMeans(n_clusters=cluster_count)
    kMeans.fit_predict(ratings[genre_labels].head(2000))
    sse_list.append(kMeans.inertia_)

plt.figure()
plt.title("SSE vs Clusters for K-Means Graph")
plt.plot(range(1,20), sse_list)
plt.xlabel('# of Clusters')
plt.ylabel('SSE for Cluster')
plt.show()

clusters = 8
kMeans = cluster.KMeans(n_clusters=clusters)
kMeans.fit(ratings[genre_labels])

labels = kMeans.labels_
ratings['cluster'] = labels

# Save to cluster, for referencing
ratings.to_csv('cluster.csv', index=False)

ratings = pd.read_csv('cluster.csv')

# Create Dataframe averaging genre scores
rf = pd.DataFrame()
rf['count'] = ratings.cluster.value_counts()
rf['percent'] = ratings.cluster.value_counts()/len(ratings)
ratingsGrouped = ratings.groupby('cluster').mean()
rf[genre_labels] = ratingsGrouped[genre_labels]
print(rf.sort_index(inplace=True))

# Create Z-Score Dataframe
rfZ = pd.DataFrame()
rfZ[['count', 'percent']] = rf[['count','percent']]
rfZ[genre_labels] = (rf[genre_labels] - rf[genre_labels].mean())/rf[genre_labels].std()

# Print out mean and std for each genre
for genre in genre_labels:
    print(genre,":", "Mean:", rf[genre].mean(), "Standard Deviation:", rf[genre].std())

# Print out Dataframes for Genre Scores and Z-Scores
print(rf)
print(rfZ)