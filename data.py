import pandas as pd
import numpy as np


df = pd.read_csv('Bengaluru_House_Data.csv')

# print(df.head().to_string())

# print(df.shape)

# print(df.info())

# for col in df.columns:
#     print(df[col].value_counts())
#     print("*"*20)

# print(df.isna().sum())

df.drop(columns=['area_type', 'availability', 'society', 'balcony'], inplace=True)

# print(df.describe())


df['location'] = df['location'].fillna('Sarjapur  Road')
# print(df['location'].value_counts())


df['size'] = df['size'].fillna('2 BHK')
# print(df['size'].value_counts())

df['bath'] = df['bath'].fillna(df['bath'].median())
# print(df.info())

df['bhk'] = df['size'].str.split().str.get(0).astype(int)


def convertRange(x) :
    temp = x.split('-')
    if(len(temp) == 2) :
        return ((float(temp[0]) + float(temp[1])) / 2.0)
    try :
        return float(x)
    except :
        return None

df['total_sqft'] = df['total_sqft'].apply(convertRange)


df['price_per_sqft'] = df['price'] * 100000.0 / df['total_sqft']
# print(df.head().to_string())

df['location'] = df['location'].apply(lambda x : x.strip())
location_count = df['location'].value_counts()
location_count_less_10 = location_count[location_count <= 10]
df['location'] = df['location'].apply(lambda x : 'other' if x in location_count_less_10 else x)

# print(df['location'].value_counts())

df = df[((df['total_sqft'] / df['bhk']) >= 300)]
# print(df.shape)

def remove_outliers_sqft(df) :
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location') :
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output

df = remove_outliers_sqft(df)
# print(df.describe())

def bhk_outlier_remover(df) :
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location') :
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk') :
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk') :
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5 :
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df = bhk_outlier_remover(df)
# print(df.shape)

df.drop(columns=['size', 'price_per_sqft'], inplace=True)

X = df.drop(columns=['price'])
y = df['price']

print(df.head().to_string())

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

categorical_features = ['location']
numerical_features = ['total_sqft', 'bath', 'bhk']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])

xgb_pipeline.fit(X_train, y_train)

y_pred = xgb_pipeline.predict(X_test)

print("\nXGBoost")
print("R2 Score :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))

df.to_csv('cleaned_data.csv', index=False)

import pickle

pickle.dump(xgb_pipeline, open('xgbModel.pkl', 'wb'))