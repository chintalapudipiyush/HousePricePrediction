#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Import dataset
df = pd.read_csv(r'C:\Users\91934\Downloads\archive\Participants_Data_HPP\Train.csv') #Location of dataset

# Extracting city from address
df['City'] = df['ADDRESS'].apply(lambda x: x.split(',')[-1].strip())  # Extracting the last part and removing whitespace

# Dropping the 'ADDRESS' column
df.drop(['ADDRESS'], axis=1, inplace=True)

# One-Hot Encoding
new_cols = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder()
encoded_cols = encoder.fit_transform(df[new_cols])
odf = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(new_cols))
df = pd.concat([df, odf], axis=1)
df.drop(new_cols, axis=1, inplace=True)

# Training data
X = df.drop('TARGET', axis=1)
# Training output
y = df['TARGET']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

# Model training
algo = LinearRegression()
algo.fit(X_train, y_train)

# Making predictions
predictions = algo.predict(X_test)

# Model evaluation
score = algo.score(X_test, y_test)
print(score)





