import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


df = pd.read_csv('Telco_Sales_data.csv')
print(df.head)

duplicates = df['CustomerID'].duplicated().any()
print(duplicates)

#Creating age group/categories
bins = [0,30,50,70,100]
labels = ['18-30','31-50','51-70','70+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

#Creating tenure group/categories
bin = [0, 6,12,24,48,100]
label = ['0-6 Months','6-12 Months','1-2 Years','2-4 Years','4+ Years']
df['TenureGroup'] = pd.cut(df['Tenure'], bins=bin, labels=label)

df_original = df.copy()
X = df.drop(['Churn', 'CustomerID'], axis=1)
y = df['Churn']

X_encoded = pd.get_dummies(X, drop_first=True)

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#Training Models
#CHurn Model
model_churn = RandomForestClassifier(random_state=42)
model_churn.fit(X_train, y_train)

#Total charges model
model_total = RandomForestRegressor(random_state=42)
model_total.fit(X_train, df_original.loc[X_train.index, 'TotalCharges'])


#Monthly charges model
model_monthly = RandomForestRegressor(random_state=42)
model_monthly.fit(X_train,df_original.loc[X_train.index, 'MonthlyCharges'])

#DOING PREDICTIONS
#CHURN PREDICTION
y_pred_churn = model_churn.predict(X_test)
y_prob_churn = model_churn.predict_proba(X_test)[:, 1]


#REVENUE PREDICTION
y_pred_total = model_total.predict(X_test)
y_pred_monthly = model_monthly.predict(X_test)

#EVALUATING MODEL
#Churn accuracy
accuracy = accuracy_score(y_test, y_pred_churn)
print(f"\nChurn Model Accuracy: {accuracy:.2f}")

#Revenue errors
mae_total = mean_absolute_error(df_original.loc[X_test.index, 'TotalCharges'], y_pred_total)
r2_total = r2_score(df_original.loc[X_test.index, 'TotalCharges'], y_pred_total)
print(f"Total Charges MAE: {mae_total:.2f}")
print(f"Total Charges R2: {r2_total:2f}")

#Results table

results = X_test.copy()
#Adding actual values
results['ActualChurn'] = y_test.values
results['ActualTotalCharges'] = df_original.loc[X_test.index, 'TotalCharges']
results['ActualMonthlyCharges'] = df_original.loc[X_test.index, 'MonthlyCharges']


#Adding predictions
results['PredictedChurn'] = y_pred_churn
results['ChurnProbability'] = y_prob_churn
results['PredictedTotalCharges'] = y_pred_total
results['PredictedMonthlyCharges'] = y_pred_monthly

#Convert churn labels
results['ActualChurn'] = results['ActualChurn'].map({0: 'No', 1: 'Yes'})
results['PredictedChurn'] = results['PredictedChurn'].map({0: 'No', 1: 'Yes'})
print("\n===== Sample RESULTS=====")
print(results[['ActualChurn', 'PredictedChurn', 'ChurnProbability', 'ActualTotalCharges', 'PredictedTotalCharges', 'ActualMonthlyCharges', 'PredictedMonthlyCharges']].head())

churn_counts = pd.Series(y_pred_churn).value_counts(normalize=True) * 100
print("\n=====CHURN PERCENTAGE=====")
print(f"WILL NOT CHURN: {churn_counts.get(0,0):.2f}%")
print(f"WILL CHURN: {churn_counts.get(1,0):.2f}%")

#Revenue summary

total_actual = results['ActualTotalCharges'].sum() 
total_pred = results['PredictedTotalCharges'].sum()
monthly_actual = results['ActualMonthlyCharges'].sum() 
monthly_pred = results['PredictedMonthlyCharges'].sum()
print("\n=====REVENUE SUMMARY=====")
print(f"Actual Total Revenue: ${total_actual:,.2f}")
print(f"Predicted Total Revenue: ${total_pred:,.2f}")

print(f"\nActual Monthly Revenue: ${monthly_actual:,.2f}")
print(f"Predicted Monthly Revenue: ${monthly_pred:,.2f}")




results.to_csv('predictions_results.csv', index=False)
churn_summary = pd.DataFrame({'Churn status': ['No', 'Yes'],
                              'Percentage (%)': [churn_counts.get(0,0),churn_counts.get(1,9)
                            ]})
churn_summary.to_csv('churn_summary.csv', index=False)


revenue_summary = pd.DataFrame({'Metric': ['Total Revenue', 'Monthly Revenue'],
                              'Actual': [total_actual, monthly_actual],
                              'Predicted': [total_pred, monthly_pred]
                              })
revenue_summary.to_csv('revenue_summary.csv', index=False)
df.to_csv('Telco_Sales_data_cleaned.csv', index=False)



