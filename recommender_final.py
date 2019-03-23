#Setting up prerequisites
#from numba import prange
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


print("Select Number of DataPoints to Train on: \n1: 1024 \t2: 10000 \n3: 25000 \t4: 75000 \n5: 100000 \t6: 200000\n\n")
choice = int(input())
print("\nLoading Data\n")
if (choice==1 or choice==1024):
	df1 = pd.read_csv('feasible_data_1024.txt', header = None, names = ['Cust_Id', 'Rating', 'Date'], usecols = [0,1,2])
elif (choice==2 or choice==10000):
	df1 = pd.read_csv('feasible_data_10000.txt', header = None, names = ['Cust_Id', 'Rating', 'Date'], usecols = [0,1,2])
elif (choice==3 or choice==25000):
	df1 = pd.read_csv('feasible_data_25000.txt', header = None, names = ['Cust_Id', 'Rating', 'Date'], usecols = [0,1,2])
elif (choice==4 or choice==75000):
	df1 = pd.read_csv('feasible_data_75000.txt', header = None, names = ['Cust_Id', 'Rating', 'Date'], usecols = [0,1,2])
elif (choice==5 or choice==100000):
	df1 = pd.read_csv('feasible_data_100000.txt', header = None, names = ['Cust_Id', 'Rating', 'Date'], usecols = [0,1,2])
elif (choice==6 or choice==200000):
	df1 = pd.read_csv('feasible_data_200000.txt', header = None, names = ['Cust_Id', 'Rating', 'Date'], usecols = [0,1,2])


df1['Rating'] = df1['Rating'].astype(float)
df1['Date'] = df1['Date'].astype(str)
df1['Date'] = df1['Date'].map( lambda s : (s[:4])+(s[5:7])+(s[8:]))
df1['Date'] = df1['Date'].astype(float)
print('Dataset 1 shape: {}'.format(df1.shape))
print('-Dataset examples-')
print(df1.iloc[::10000, :])
#print(df1['Date'].dtype)
df = df1




#Seeing the distribution of ratings given by the users
#print("See Overview of the Data")
p = df.groupby('Rating')['Rating'].agg(['count'])
# get movie count
movie_count = df.isnull().sum()[1]
# get customer count
cust_count = df['Cust_Id'].nunique() - movie_count
# get rating count
rating_count = df['Cust_Id'].count() - movie_count
ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
plt.axis('off')
for i in range(1,6):
    ax.text(p.iloc[i-1][0]/4, i-1, 'Rated {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')










#Adding movie IDs to the dataset
print("\nExtracting Movie IDs\n")
movie_np = []
movie_id = 0
for x in range(df.shape[0]):
    if(np.isnan(df.iloc[x]['Rating'])):
        movie_id = movie_id+1
    movie_np = np.append(movie_np,movie_id)
#print(movie_np)
#print(len(movie_np))
df['Movie_Id'] = movie_np.astype(int)
print("Movie IDs extracted from the extra rows given")








# remove the extra Movie ID rows
print("\nRemoving extra Movie ID rows\n")
df = df[pd.notnull(df['Rating'])]
df['Cust_Id'] = df['Cust_Id'].astype(int)
print('-Dataset examples-')
print(df.iloc[::100, :])
print("\n\nThese are the final datatypes of the dataset")
print(df.dtypes)




#Creating Data Matrix
df_matrix=pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')
print(df_matrix.shape)



#Loading the Movie ID- Movie Title Mapping File
print("\nLoading the Movie ID- Movie Title Mapping File\n")
df_title = pd.read_csv('netflix-prize-data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)
print("See some Movie ID- Movie Title Mapping : \n")
print (df_title.head(8))




print("\n\nData Cleaning Complete.\n See head of the Data Matrix:\n")
print(df_matrix.head())
n_movies = movie_count
n_customers = cust_count
print("\nNum of movies =", movie_count)
print("Num of users =", cust_count)
print()








#Choosing the number of latent attributes
n_attr= 100*1000000
#print(type(n_attr),type(n_movies), type(n_customers))
Q = Variable((n_attr,n_movies))
P = Variable((n_attr, n_customers))
acq_data = df_matrix.fillna(0.0)
print(acq_data.head())








R = np.array(acq_data)
R1= np.array(acq_data)


print("\nRandomly Distributing Test and Train Set by removing 20% values...\n")
#This cell works on Real DataSet
R = np.array(acq_data)
R1= np.array(acq_data)
#Set the number of values to replace. For example 20%:
prop = int(R.size * 0.2)
#Randomly choose indices of the numpy array:
#print("Creating Random Indices\n")
# i = [np.random.choice(range(R.shape[0])) for _ in range(prop)]
# j = [np.random.choice(range(R.shape[1])) for _ in range(prop)]
i = np.random.randint(0,R.shape[0],size=prop)
j = np.random.randint(0,R.shape[1],size=prop)
#print("Created Random Indices\n")
print("Done\n")
#print("i=",i)
#print("j=",j)
#Change values with 0
R[i,j] = 0
print("Original:\n",R1)
print("Test Set:\n",R)
R=np.rint(R)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(R, R1)
print("RMSE=",mse**0.5)
print("\nTraining ...\n")
mf = MF(R, K=2, alpha=0.01, beta=0.01, iterations=100)
training_process = mf.train()
L=np.rint(mf.full_matrix())
print("\nDone\n")
x = [x for x, y in training_process]
y = [y for x, y in training_process]
x = x[::10]
y = y[::10]
plt.figure(figsize=((16,4)))
plt.plot(x, np.sqrt(y))
plt.xticks(x, x)
print("Minimizing Error on Training Set:\n")
plt.xlabel("Iterations")
plt.ylabel("Root Mean Square Error")
plt.grid(axis="y")
print("Learnt=\n",mf.full_matrix())
print("\nRating predictions=\n",L)
print()
print()
# print("Global bias:")
# print(mf.b)
# print()
# print("User bias:")
# print(mf.b_u)
# print()
# print("Item bias:")
# print(mf.b_i)
print("\nFinding Error on test set...\n")
msef=0.0
# for i1 in range(len(i)):
#     for i2 in range(len(j)):
#         if R1.item(i[i1],j[i2])!=0:
#             msef = msef + (R1.item((i[i1],j[i2]))-(L).item((i[i1],j[i2])))**2
# msef = (msef/(len(j)*len(i)))
valid_cmp = ~np.isnan(df_matrix)
msef = np.sum(np.sum(np.multiply(valid_cmp,np.square(R1-L)),axis=None))/(len(j)*len(i)*1.00)

print("RMSE final=",msef**0.5)