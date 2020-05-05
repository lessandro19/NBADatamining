"""
Sergio Paniagua
1001523347
"""


#import tools and libraries

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#read our csv files containing the stats for each nba player
nba = pd.read_csv('NBAstats.csv')

#define our classification column and the features/attributes we will use 
class_column = 'Pos'
feature_columns = ['G', '3P', '3PA', \
     '2P', '2PA', 'eFG%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'PF']

#get the relevant nba features from our data
nba_feature = nba[feature_columns]

#get the relevant classifications from our data
nba_class = nba[class_column]

#create a trainingfeature,training class, test feature, and test class

train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25,random_state=0)

#Create a decision tree and make the maximum amount of levels limited to 9
tree = DecisionTreeClassifier(max_depth=9,random_state=0)
#use our tree on our data
tree.fit(train_feature, train_class)

#print the results of our training and test data
print("Training set score: {:.3f}".format(tree.score(train_feature, train_class)))
print("Test set score: {:.3f}".format(tree.score(test_feature, test_class)))

#create the confusion matrix for our data
prediction = tree.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

#calculate the score of our 10 fold stratified cross validation
scores=cross_val_score(tree,nba_feature,nba_class,cv=10)

#print our 10 cross validation scores and the average cross
#validation score
print("\nCross-validation scores: {})".format(scores))
print("\nAverage cross-validation score: {:.2f}".format(scores.mean()))
