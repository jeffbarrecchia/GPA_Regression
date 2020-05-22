#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:35:31 2020

@author: jeffbarrecchia
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split as ttSplit
from sklearn.linear_model import LinearRegression

df = pd.read_csv('~/Documents/Kaggle_Projects/student-mat.csv')

# =============================================================================
# Tells how many individual students there are
# =============================================================================

# print('There are:', len(df['sex']), 'different students.')

# =============================================================================
# Distribution graphs
# =============================================================================

# grade_dist = sb.countplot(df['G3'])
# sex_dist = sb.countplot(df['age'], hue = df['sex'])
# plt.title('Grade 1 to Grade 2 Comparison')
# sb.scatterplot(df['G1'], df['G2'])
# plt.title('Grade 1 to Grade 3 Comparison')
# sb.scatterplot(df['G1'], df['G3'])
# plt.title('Grade 2 to Grade 3 Comparison')
# sb.scatterplot(df['G2'], df['G3'])
# sb.distplot(df['G1'], hist = False, label = 'G1')
# sb.distplot(df['G2'], hist = False, label = 'G2')
# sb.distplot(df['G3'], hist = False, label = 'G3')
# plt.xlabel('Score')
# plt.ylabel('Percentage')
# plt.title('Distribution of Grades')
# sb.plt.show()

# =============================================================================
# Turns the categorical variables into numerical for Regression purposes
# =============================================================================

cleaned_df = pd.get_dummies(df, columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])

# =============================================================================
# Finds correlation between all variables and makes a heatmap for it
# =============================================================================

# corr = cleaned_df.corr()
# plt.figure(figsize = (20, 20))
# sb.heatmap(corr, annot = True, annot_kws = {'size': 5}, cmap = 'Blues')

# =============================================================================
# Preps variables for Regression by dividing them into x and y
# =============================================================================

df_x = cleaned_df.drop(columns = ['G3'])
df_y = cleaned_df['G3']

# =============================================================================
# Splits dataset into a training and test dataset
# =============================================================================

x_train, x_test, y_train, y_test = ttSplit(df_x, df_y, train_size = 0.8, random_state = 4)
list_y_test = list(y_test)

# =============================================================================
# Performs multi-variate Regression
# =============================================================================

regr = LinearRegression()

regr.fit(x_train, y_train)

predict = regr.predict(x_test)

# =============================================================================
# Gets the accuracy of the regression for the train and test dataset
# =============================================================================

acc_train = regr.score(x_train, y_train)
acc_test = regr.score(x_test, y_test)

print('\nThe accuracy of the training dataset is: {:.2f}'.format(acc_train * 100) + '%')
print('\nThe accuracy of the test dataset is: {:.2f}'.format(acc_test * 100) + '%')









