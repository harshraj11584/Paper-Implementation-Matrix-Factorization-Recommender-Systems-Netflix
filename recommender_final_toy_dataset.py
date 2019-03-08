#Setting up prerequisites
from mf import MF
import pandas as pd
import numpy as np
import math
import re
import sklearn
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD, evaluate
sns.set_style("darkgrid")
from cvxpy import *
from numpy import matrix
print("Setup Complete\n")


# df1 = pd.read_csv('netflix-prize-data/toy_combined_data.txt', header = None, names = ['Cust_Id', 'Rating', 'Date'], usecols = [0,1,2])
# df1['Rating'] = df1['Rating'].astype(float)
# df1['Date'] = df1['Date'].astype(str)
# df1['Date'] = df1['Date'].map( lambda s : (s[:4])+(s[5:7])+(s[8:]))
# df1['Date'] = df1['Date'].astype(float)
# print('Dataset 1 shape: {}'.format(df1.shape))
# print('-Dataset examples-')
# print(df1.iloc[::100, :])
# print(df1['Date'].dtype)
# df = df1






# #Seeing the distribution of ratings given by the users
# print("See Overview of the Data")
# p = df.groupby('Rating')['Rating'].agg(['count'])
# # get movie count
# movie_count = df.isnull().sum()[1]
# # get customer count
# cust_count = df['Cust_Id'].nunique() - movie_count
# # get rating count
# rating_count = df['Cust_Id'].count() - movie_count
# ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
# plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
# plt.axis('off')
# for i in range(1,6):
#     ax.text(p.iloc[i-1][0]/4, i-1, 'Rated {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')










# #Adding movie IDs to the dataset
# movie_np = []
# movie_id = 0
# for x in range(df.shape[0]):
#     if(np.isnan(df.iloc[x]['Rating'])):
#         movie_id = movie_id+1
#     movie_np = np.append(movie_np,movie_id)
# #print(movie_np)
# #print(len(movie_np))
# df['Movie_Id'] = movie_np.astype(int)
# print("Movie IDs extracted from the extra rows given")








# # remove the extra Movie ID rows
# df = df[pd.notnull(df['Rating'])]
# df['Cust_Id'] = df['Cust_Id'].astype(int)
# print('-Dataset examples-')
# print(df.iloc[::100, :])
# print("\n\nThese are the final datatypes of the dataset")
# print(df.dtypes)




# #Creating Data Matrix
# df_matrix=pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')
# print(df_matrix.shape)



# #Loading the Movie ID- Movie Title Mapping File
# df_title = pd.read_csv('netflix-prize-data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
# df_title.set_index('Movie_Id', inplace = True)
# print("See some Movie ID- Movie Title Mapping : \n")
# print (df_title.head(8))




# print("\n\nData Cleaning Complete.\n See head of the Data Matrix:\n")
# print(df_matrix.head())
# n_movies = movie_count
# n_customers = cust_count
# print("\nNum of movies =", movie_count)
# print("Num of users =", cust_count)









# #Choosing the number of latent attributes
# n_attr= 100*1000000
# #print(type(n_attr),type(n_movies), type(n_customers))
# Q = Variable((n_attr,n_movies))
# P = Variable((n_attr, n_customers))
# acq_data = df_matrix.fillna(0.0)
# print(acq_data.head())





#This cell works on Toy Dataset
#The next cell is for real data
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
R1= np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
#Set the number of values to replace. For example 20%:
prop = int(R.size * 0.2)
#Randomly choose indices of the numpy array:
i = [np.random.choice(range(R.shape[0])) for _ in range(prop)]
j = [np.random.choice(range(R.shape[1])) for _ in range(prop)]
#Change values with 0
R[i,j] = 0
print("Original:\n",R1)
print("Test Set:\n",R)
R=np.rint(R)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(R, R1)
print("MSE=",mse**0.5)
print("\nTraining ...\n")
mf = MF(R, K=10000, alpha=0.01, beta=0.01, iterations=10000)
training_process = mf.train()
L=np.rint(mf.full_matrix())
print("Learnt=\n",L)
print("\nFinding Error on test set...\n")
msef=0.0
for i1 in range(len(i)):
    for i2 in range(len(j)):
        if R1.item(i[i1],j[i2])!=0:
            msef = msef + (R1.item((i[i1],j[i2]))-(L).item((i[i1],j[i2])))**2
msef = (msef/(len(j)*len(i)))
print("RMSE f=",msef**0.5)