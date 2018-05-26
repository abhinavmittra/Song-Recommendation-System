import pandas as pd
from sklearn.model_selection import train_test_split

#user_id,song_id,listening time
three_features_file = 'https://static.turi.com/datasets/millionsong/10000.txt'  
#song_id,title,released_by,artist_name
songsmetadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv' 


songs_df1 = pd.read_table(three_features_file,header = None)
songs_df1.colums = ['user_id','song_id','listen_count']
songs_df2 = pd.read_csv(songsmetdata_file)

#Merge dataframes 
songs_df = pd.merge(songs_df1,songs_df2.drop_duplicates(['song_id']),on='song_id',how = 'left')

#Grouping according to listen_count and then calculating percentages of each
songs_grouped = songs_df.groupby(by = ['song']).agg({'listen_count:count'}).reset_index()
grouped_sum = songs_grouped['listen_count'].sum()
songs_grouped['percentage'] = songs_grouped.div(grouped_sum)*100
songs_grouped.sort_values(['listen_count','song'],ascending = [0,1])  

#Number of unique songs and users
users = songs_df['user_id'].unique()
songs = songs_df['song_id'].unique()

train_set,test_set = train_test_split(songs_df,test_size = 0.2,random_state = 10)

#Using a popularity based Recommender as a blackbox to train the model (Naive Based Approach)
#gets a unique count of the user_id for each song and tags it with a score
#recommend accepts a user_id and outputs the top ten recommended songs for that user

pop = Recommender.popularity_recommender_py()
pop.create(train_set,'user_id','song')
pop.recommend(user_id)
