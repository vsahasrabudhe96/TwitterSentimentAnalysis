from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from preprocessing import preprocessing
import pandas as pd
from sklearn.metrics import silhouette_score
import pickle

def clustering(df,col):
    df = preprocessing(df,col)
    
    vectorizer = TfidfVectorizer(stop_words="english",sublinear_tf=True,min_df=5,max_df=0.95)
    X = vectorizer.fit_transform(df[col])
    
    kmeans = KMeans(n_clusters=3,random_state=42,init='k-means++',max_iter=100,n_init=1)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    # print(c1)
    # clusters = kmeans.labels_
    # print(clusters)
    
    df['cluster'] = clusters
    pca = PCA(n_components=2,random_state=42)

    pca_vecs = pca.fit_transform(X.toarray())

    x0 = pca_vecs[:,0]
    x1 = pca_vecs[:,1]
    df['x0'] = x0
    df['x1'] = x1

    cluster_map = {0:'Negative',1:"Positive",2:"Neutral"}

    df['cluster'] = df['cluster'].map(cluster_map)
    
    sc = silhouette_score(X,labels=kmeans.predict(X))
    pickle.dump(kmeans, open("../model/save.pkl", "wb"))
    
    return df