import pandas as pd
import numpy as np
import visualize_support as vz
import feature_engineer_support as fes
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
import model_support as ms
import re as re
from sklearn import preprocessing

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
	ticket = ticket.replace('.','')
	ticket = ticket.replace('/','')
	ticket = ticket.split()
	ticket = map(lambda t : t.strip(), ticket)
	ticket = list(filter(lambda t : not t.isdigit(), ticket))
	#    print("-----------",ticket)
	if len(ticket) > 0:
	    return ticket[0]
	else: 
	    return 'XXX'

def feature_engineer(train, test):
    full_data = [train, test]

    for dataset in full_data:
        dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
  
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    
    for dataset in full_data:
        age_avg        = dataset['Age'].mean()
        age_std        = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)

    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)

    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    embark_dummies = []
    title_dummies = []

    # Data Cleaning
    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
        
        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        title_dummies.append(pd.get_dummies(dataset['Title'], prefix='Title'))
        dataset.drop('Title', axis=1)

        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
        embark_dummies.append(pd.get_dummies(dataset['Embarked'], prefix='Embarked'))
        dataset.drop('Embarked', axis=1)

        # Mapping Fare
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
        
        # Mapping Age
        dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
    train = pd.concat([embark_dummies[0], title_dummies[0], train], axis=1)
    test = pd.concat([embark_dummies[1], title_dummies[1], test], axis=1)

	# Feature Selection
    drop_elements = ['PassengerId', 'Name',\
                      'Cabin', 'Ticket']
    
    train = train.drop(drop_elements, axis = 1)
    test  = test.drop(drop_elements, axis = 1)

    return train, test

def predict(model, test, combined):
    output = model.predict(test).astype(int)
    df_output = pd.DataFrame()
    df_output['PassengerId'] = combined['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

# load data
train = pd.read_csv('train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})

# feature engineering
train_new, test_new = feature_engineer(train, test)
targets = train_new["Survived"]
train_new = train_new.drop(["Survived"], axis = 1)

# normalize features
features = list(train_new.columns.values)
min_max_scaler = preprocessing.MinMaxScaler()

train_new = min_max_scaler.fit_transform(train_new)
train_new = pd.DataFrame(train_new, columns = features)

test_new = min_max_scaler.fit_transform(test_new)
test_new = pd.DataFrame(test_new, columns = features)

print train_new.head()

# train model
#model = ms.train_naivebayes(train_new, targets) # Val score: 0.76, Test Score: 0.72249
model = ms.train_model_randomforest(train_new, targets) # Val score: 0.836139169473, Test Score: 0.79904
#model = ms.train_model_gradientboost(train_new, targets) # Val score: 0.836139169473, Test Score: 0.77033
#model = ms.train_model_adaboost(train_new, targets) # Val score: 0.821548821549, Test Score: 0.74641
#model = ms.train_model_lda(train_new, targets) # Val score: 0.828282828283, Test Score: 0.77512
#model = ms.train_model_svm(train_new, targets) # Val score: 0.828282828283, Test Score: 0.77512
#model = ms.train_model_xgboost(train_new, targets) # Val score: 0.828282828283, Test Score: 0.77512
#model = ms.train_ensemble(train_new, targets)
predict(model, test_new, test)