import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

def getGamma(sigma):

        # return sigma value
        return 1.0/(2*(sigma**2))

def train_naivebayes(train, targets):
    forest = GaussianNB()

    parameter_grid = {}

    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    print('Best score Naive Bayes: {}'.format(grid_search.best_score_))
    print('Best parameters Naive Bayes: {}'.format(grid_search.best_params_))
    return grid_search

def train_model_randomforest(train, targets):
    forest = RandomForestClassifier()

    parameter_grid = {
                     'max_features': [1.0],
                     'max_depth' : [6,7,8],
                     'n_estimators': range(70,110,10)
                     }

    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    # joblib.dump(grid_search, 'randomforest5.pkl', compress=9)
    # grid_search = joblib.load('randomforest.pkl')
    print('Best score Random Forest: {}'.format(grid_search.best_score_))
    print('Best parameters Random Forest: {}'.format(grid_search.best_params_))
    return grid_search

def train_model_gradientboost(train, targets):
    forest = GradientBoostingClassifier()

    parameter_grid = {
                     'max_features': [1.0],
                     'max_depth' : [2,3,4,5],
                     'n_estimators': range(70, 120, 10)
                     }

    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    print('Best score Gradient Boost: {}'.format(grid_search.best_score_))
    print('Best parameters Gradient Boost: {}'.format(grid_search.best_params_))
    return grid_search

def train_model_adaboost(train, targets):
    forest = AdaBoostClassifier()

    parameter_grid = {
                     'learning_rate': [0.01, 0.1, 1],
                     'n_estimators': range(100, 160, 10)
                     }

    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    print('Best score ADA Boost: {}'.format(grid_search.best_score_))
    print('Best parameters ADA Boost: {}'.format(grid_search.best_params_))
    return grid_search

def train_model_lda(train, targets):
    forest = LinearDiscriminantAnalysis()

    parameter_grid = {
                        'n_components': range(1,10,1)
                     }

    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    print('Best score LDA: {}'.format(grid_search.best_score_))
    print('Best parameters LDA: {}'.format(grid_search.best_params_))
    return grid_search

def train_model_svm(train, targets):
    forest = svm.SVC()
    gammas = [getGamma(10**0), getGamma(10**1), getGamma(10**2), getGamma(10**3), getGamma(10**4)]

    parameter_grid = {'C': [10**-3, 10**-2, 10**10], 
                      'gamma': gammas,
                    'max_iter' : [10000]}

    cross_validation = StratifiedKFold(targets, n_folds=10)

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    print('Best score SVM: {}'.format(grid_search.best_score_))
    print('Best parameters SVM: {}'.format(grid_search.best_params_))
    return grid_search

def train_model_xgboost(train, targets):
    forest = XGBClassifier()
    
    parameter_grid = {'max_depth': [3, 10], 
                      'learning_rate': [0.01, .1, 1],
                    'n_estimators' : [10, 100, 1000]}


    cross_validation = StratifiedKFold(targets, n_folds=10)

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    print('Best score xgboost: {}'.format(grid_search.best_score_))
    print('Best parameters xgboost: {}'.format(grid_search.best_params_))
    return grid_search

def train_ensemble(train, targets):
    estimators = []
    modelNB = GaussianNB()
    estimators.append(('naive bayes', modelNB))
    modelRF = RandomForestClassifier(max_features=1.0, n_estimators=80, max_depth=8)
    estimators.append(('random forest', modelRF))
    ensemble = VotingClassifier(estimators)
    results = model_selection.cross_val_score(ensemble, train, targets, cv=kfold)
    print(results.mean())