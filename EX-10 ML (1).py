#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('Mall_Customers.csv')
data


# In[21]:


data.head()


# In[3]:


data.isnull().sum()


# In[22]:


from sklearn.cluster import KMeans
wcss=[]
print("Name:A.Lahari")
print("Reg.No:212223230111")


# In[19]:


for i in range(1,11):
    Kmeans= KMeans(n_clusters = i,init="k-means++")
    Kmeans.fit(data.iloc[:,3:])
    wcss.append(Kmeans.inertia_)


# In[6]:


import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.xlabel("No.of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")


# In[17]:


km=KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])


# In[8]:


y_pred=km.predict(data.iloc[:,3:])
y_pred


# In[15]:


data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="pink",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")


# In[ ]:




