''' Clustering bank customers using K-Means clustering '''

# import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# get all our files
customers = pd.read_csv('twm_customer.csv', delimiter=';')

# drop unneeded cols from customers
df1 = customers.drop(columns=['cust_id', 'name_prefix', 'first_name', 
'last_name', 'street_nbr', 'street_name', 'postal_code', 'city_name', 'state_code'])

#one hot encoding for marital status and gender
df1 = pd.concat([df1.drop('marital_status', axis=1), pd.get_dummies(df1['marital_status'], prefix='MaritalStatus')], axis=1)
df1 = pd.concat([df1.drop('gender', axis=1), pd.get_dummies(df1['gender'], prefix='gender')], axis=1)

# scale the data
scaler = MinMaxScaler()
df2 = scaler.fit_transform(df1)

# do PCA on the data
pca = PCA(n_components=2)

# transform the data
df3 = pca.fit_transform(df2)

# create kmeans model
km = KMeans(n_clusters=6, n_init=10, random_state=0)

# fit and predict
label = km.fit_predict(df3)

# make dataframe out of scaled data for radar chart
df4 = pd.DataFrame(df2, columns=df1.columns)

# add cluster label to df
df4['cluster'] = label

# aggregate by cluster
dfradar = df4.groupby('cluster').mean()

# categorie names
categories = list(dfradar.columns)

# make radar charts for each cluster
fig = go.Figure()

for i in range(0,4):
    fig.add_trace(go.Scatterpolar(
        r=dfradar.iloc[i,:],
        theta=categories,
        fill='toself',
        name=f'Cluster {i}'
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0,5]
        )),
    showlegend=False
)

# get accounts and transactions
accounts = pd.read_csv('twm_accounts.csv', delimiter=';')

transactions = pd.read_csv('twm_transactions.csv', delimiter=';')

joint_df = pd.merge(accounts, transactions, on='acct_nbr', how='outer')

# tran_date is number of days between tran_date and acct_start_date
joint_df['tran_date'] = (pd.to_datetime(joint_df['tran_date']) - pd.to_datetime(joint_df['acct_start_date'])).dt.days


# hot encode acct_type, account_active, channel, tran_code
hotenc = ['acct_type', 'account_active', 'channel', 'tran_code']

for i in hotenc:
    joint_df = pd.concat([joint_df.drop(i, axis=1), pd.get_dummies(joint_df[i], prefix=i)], axis=1)

# remove cust_id column
joint_df = joint_df.drop(columns=['cust_id', 'acct_start_date', 'acct_end_date'])

# fill na values
joint_df.fillna(0, inplace=True)

# scale the data
scaler = MinMaxScaler()
joint_df = scaler.fit_transform(joint_df)

