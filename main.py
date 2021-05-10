#%%
import pandas as pd
import numpy as np

# read in all our data
star_data = pd.read_csv("Stars.csv")
len(star_data)
# %%
star_data.head()
# %%
missing_values_count = star_data.isnull().sum()

#%%
from mlxtend.preprocessing import minmax_scaling

import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Temperature Scaling
original_data = pd.DataFrame(star_data.Temperature)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=['Temperature'])

# plot both together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")

star_data.Temperature = minmax_scaling(pd.DataFrame(star_data.Temperature), columns=['Temperature'])
# %%
# R, L and A_M Scaling
star_data.R = minmax_scaling(pd.DataFrame(star_data.R), columns=['R'])
star_data.L = minmax_scaling(pd.DataFrame(star_data.L), columns=['L'])
star_data.A_M = minmax_scaling(pd.DataFrame(star_data.A_M), columns=['A_M'])

# %%
# Spectral_Class Parsing
sc_to_num = {'O':0.0, 'B':0.167,'A': 0.333,'F': 0.5,'G': 0.666,'K': 0.833,'M': 1.0}

star_data['Spectral_Class'] = star_data['Spectral_Class'].apply(lambda mark: sc_to_num[mark])
# %%
# Splitting to Test and Lern sets
lables = star_data.pop('Type').values
star_data.pop('Color').values
from sklearn.model_selection import train_test_split
Star_train, Star_test, SLables_train, SLables_test = train_test_split(star_data, lables, test_size=0.33, random_state=0)
# %%
# sklearn KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
clf.fit(Star_train, SLables_train)
# %%
# Scoring
clf.score(Star_test, SLables_test)
# %%
# KNN Self Implimentation
import KnnClassifier
from KnnClassifier import KNNClassifier

sclf = KNNClassifier(5)
sclf.fit(Star_train, SLables_train)
sclf.score(Star_test, SLables_test)


# %%
# sklearn GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(Star_train, SLables_train)
gnb.score(Star_test, SLables_test)
# %%
# GaussianNB Self Implimentation
from GNBClassifier import GNBClassifier

sgnb = GNBClassifier()
sgnb.fit(Star_train, SLables_train)
sgnb.score(Star_test, SLables_test)

# %%
