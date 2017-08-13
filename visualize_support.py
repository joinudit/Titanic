# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier

def vizualize_age_survival(data):
    figure = plt.figure(figsize=(13,8))
    plt.hist([data[data['Survived']==1]['Age'].fillna(data.Age.mean()),
                data[data['Survived']==0]['Age'].fillna(data.Age.mean())], 
                stacked=False, color = ['g','r'],
                bins = 30,label = ['Survived','Dead'])
    plt.xlabel('Age')
    plt.ylabel('Number of passengers')
    plt.legend()
    plt.show()

def vizualize_fare_survival(data):
    figure = plt.figure(figsize=(13,8))
    plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=False, color = ['g','r'],
             bins = 30,label = ['Survived','Dead'])
    plt.xlabel('Fare')
    plt.ylabel('Number of passengers')
    plt.legend()
    plt.show()

def vizualize_age_fare_survival(data):
    plt.figure(figsize=(13,8))
    ax = plt.subplot()
    ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40)
    ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)
    ax.set_xlabel('Age')
    ax.set_ylabel('Fare')
    ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
    plt.show()
    
def visualize_category(data, col):
    survived_sex = data[data['Survived']==1][col].value_counts()
    dead_sex = data[data['Survived']==0][col].value_counts()
    df = pd.DataFrame([survived_sex,dead_sex])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(13,8))
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    sns.plt.show()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()
    sns.plt.show()

def plot_correlation_map( df ):
    corr = df.corr()
    f, ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
    sns.plt.show()

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )

def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    plt.show()
    print (model.score( X , y ))