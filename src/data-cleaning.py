# import libraries
import pandas as pd 
import numpy as np

# inspect the data
df = pd.read_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\Titanic-survival-rate\\data\\titanic.csv')
print(df.head())

# duplicates 
duplicates = df.duplicated().sum()
print(f'Number of duplicate rows: {duplicates}')

# missing values
missing_values = df.isnull().sum()
print('Missing values in each column:')
print(missing_values)

# fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

null_counts = df.isnull().sum()
columns_with_nulls = null_counts[null_counts > 0]
print(columns_with_nulls)

# working on cabin 
# print(df['Cabin'].unique())

# create a new feature for cabin presence
df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
# df.drop('Cabin', axis=1, inplace=True)
print(df['Has_Cabin'].value_counts())

# FINAL CHECK
print('Final check for missing values:')
print(df.isnull().sum())

# WORKING ON EMBARKED
print(df['Embarked'].value_counts())
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print('Missing values in Embarked after filling:')
print(df['Embarked'].isnull().sum())

# Calculate IQR bounds for Age (k=1.5)
Q1_age = df['Age'].quantile(0.25)
Q3_age = df['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
upper_age_bound = Q3_age + 1.5 * IQR_age
lower_age_bound = Q1_age - 1.5 * IQR_age

# Apply Capping (Winsorization)
# Cap values to the calculated bounds (lower bound minimum is 0)
df['Age_capped'] = df['Age'].clip(
    lower=max(0, lower_age_bound),
    upper=upper_age_bound
)

print('Age statistics after capping:')
print(df['Age_capped'].head())

# Transforming the fare column
# transforming data like Fare which is strictly non-negative 
# and contains zero values, log1p(fare) is the safest 
# and most numerically robust choice.
df['Fare_log'] = np.log1p(df['Fare']) # Try the IQR later if necessary
print('Fare statistics after log transformation:')
print(df['Fare_log'].head())

# Encoding categorical variables
# Encoding Sex
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
print('Sex encoding:')
print(df[['Sex', 'Sex_encoded']].head())

# Encoding Embarked using one-hot encoding
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
df = pd.concat([df, embarked_dummies], axis=1)
print('Embarked encoding:')
print(embarked_dummies.head())

# Apply One-Hot Encoding for Embarked
# df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=True)
# print('Embarked encoding:')
# print(df.filter(like='Embarked_').head())

# Extract the first letter (Deck) from the Cabin string
df['Deck'] = df['Cabin'].str[0]

# Apply One-Hot Encoding for Deck
df = pd.get_dummies(df, columns=['Deck'], prefix='Deck', drop_first=True)
print('Deck encoding:')
print(df.filter(like='Deck_').head())

print(df.dtypes)
# Final dataframe check
print('Final dataframe columns:')
print(df.columns)

# convert boolean to int
# df['Has_Cabin'] = df['Has_Cabin'].astype(int)
bool_cols = df.select_dtypes(include='bool').columns
for col in bool_cols:
    df[col] = df[col].astype(int)
print(df.dtypes)

# sex encoding
df['Sex_Encoded'] = df['Sex'].map({'male': 0, 'female': 1})

# age capping to int 
df['Age_capped'] = df['Age_capped'].astype(int)
print(df.dtypes)

# dropping unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'Age', 'Fare',"Sex_Encoded"], axis=1, inplace=True)
print('Columns after dropping unnecessary ones:')
print(df.columns)

# dtypes check
# print(df.dtypes)

# drop columns
df.drop(['Embarked_Q',"Embarked_S","Deck_B","Deck_C","Deck_D","Deck_E","Deck_F","Deck_G","Deck_T"], axis=1, inplace=True)
# save cleaned data
df.to_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\Titanic-survival-rate\\data\\titanic_cleaned_new.csv', index=False)