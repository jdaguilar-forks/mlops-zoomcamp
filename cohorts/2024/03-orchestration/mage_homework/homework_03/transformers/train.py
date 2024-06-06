import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def make_training(df):
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    print(f'Feature matrix size: {X_train.shape}')
    
    target = 'duration'
    y_train = df[target].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    print(f'Train RMSE: {mean_squared_error(y_train, y_pred, squared=False)}')
    print(f'intercept_: {lr.intercept_}')

    return dv, lr