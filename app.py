from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
import numpy as np
import Popularity_Recommender
import Song_Recommender


app = Flask(__name__)

#data
song_df_1 = pd.read_csv('triplets_file.csv') 

song_df_2 = pd.read_csv('song_data.csv') 

song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on='song_id', how='left') 

# creating new feature combining title and artist name
song_df['song'] = song_df['title']+' - '+song_df['artist_name'] 

# taking top 10k samples for quick results
song_df = song_df.head(10000) 

# cummulative sum of listen count of the songs
song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index() 

grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = (song_grouped['listen_count'] / grouped_sum ) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])
# data ends


@app.route("/index.html",methods=["POST","GET"])
def home():
    if request.method == "POST":
        user_index = request.form['user']
        user_index = int(user_index)
        #Popularity recommender start
        pr = Popularity_Recommender.popularity_recommender()
        pr.create(song_df, 'user_id', 'song') 
        # display the top 5 popular songs
        pop_recommendations = pr.recommend(song_df['user_id'][user_index])
        #pop_recommendations = Recommends top 5 popular song to the user
        #Collaborative recommendation
        ir = Song_Recommender.song_similarity_recommender()
        ir.create(song_df, 'user_id', 'song')
        collab_recommendations = ir.recommend(song_df['user_id'][user_index])
        return render_template("index.html",pop_recommnd = pop_recommendations, collab_recommnd = collab_recommendations)
    return render_template("index.html",pop_recommnd=[], collab_recommnd = [])

@app.route("/search.html",methods=["POST","GET"])
def search():
    if request.method == "POST":
        ir = Song_Recommender.song_similarity_recommender() 
        ir.create(song_df, 'user_id', 'song')
        entered_songs =  request.form['song'].split(';')
        content_recommendations = ir.get_similar_songs(entered_songs)
        return render_template("search.html", content_recommnd = content_recommendations)
    return render_template("search.html", content_recommnd = [])

if __name__ == "__main__":
    app.run()