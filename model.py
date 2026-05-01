df_original = df.copy()

# Features for churn model
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# Encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# ============================================
# 6. TRAIN MODELS
# ============================================

# ---- CHURN MODEL ----
model_churn = RandomForestClassifier(random_state=42)
model_churn.fit(X_train, y_train)

# ---- TOTAL CHARGES MODEL ----
model_total = RandomForestRegressor(random_state=42)
model_total.fit(X_train, df_original.loc[X_train.index, 'TotalCharges'])

# ---- MONTHLY CHARGES MODEL ----
model_monthly = RandomForestRegressor(random_state=42)
model_monthly.fit(X_train, df_original.loc[X_train.index, 'MonthlyCharges'])

# ============================================
# 7. PREDICTIONS
# ============================================

# Churn predictions
y_pred_churn = model_churn.predict(X_test)
y_prob_churn = model_churn.predict_proba(X_test)[:, 1]

# Revenue predictions
y_pred_total = model_total.predict(X_test)
y_pred_monthly = model_monthly.predict(X_test)

# ============================================
# 8. MODEL EVALUATION
# ============================================

# Churn accuracy
accuracy = accuracy_score(y_test, y_pred_churn)
print(f"\nChurn Model Accuracy: {accuracy:.2f}")

# Revenue errors
mae_total = mean_absolute_error(df_original.loc[X_test.index, 'TotalCharges'], y_pred_total)
r2_total = r2_score(df_original.loc[X_test.index, 'TotalCharges'], y_pred_total)

print(f"Total Charges MAE: {mae_total:.2f}")
print(f"Total Charges R2: {r2_total:.2f}")

# ============================================
# 9. RESULTS TABLE
# ============================================

results = X_test.copy()

# Add actual values
results['ActualChurn'] = y_test.values
results['ActualTotalCharges'] = df_original.loc[X_test.index, 'TotalCharges']
results['ActualMonthlyCharges'] = df_original.loc[X_test.index, 'MonthlyCharges']

# Add predictions
results['PredictedChurn'] = y_pred_churn
results['ChurnProbability'] = y_prob_churn
results['PredictedTotalCharges'] = y_pred_total
results['PredictedMonthlyCharges'] = y_pred_monthly

# Convert churn labels
results['ActualChurn'] = results['ActualChurn'].map({0: 'No', 1: 'Yes'})
results['PredictedChurn'] = results['PredictedChurn'].map({0: 'No', 1: 'Yes'})

print("\n===== SAMPLE RESULTS =====")
print(results[['ActualChurn', 'PredictedChurn', 'ChurnProbability',
               'ActualTotalCharges', 'PredictedTotalCharges',
               'ActualMonthlyCharges', 'PredictedMonthlyCharges']].head())

# ============================================
# 10. CHURN PERCENTAGE
# ============================================

churn_counts = pd.Series(y_pred_churn).value_counts(normalize=True) * 100

print("\n===== CHURN PERCENTAGE =====")
print(f"Will NOT churn: {churn_counts.get(0,0):.2f}%")
print(f"Will churn: {churn_counts.get(1,0):.2f}%")

# ============================================
# 11. REVENUE SUMMARY
# ============================================

total_actual = results['ActualTotalCharges'].sum()
total_pred = results['PredictedTotalCharges'].sum()

monthly_actual = results['ActualMonthlyCharges'].sum()
monthly_pred = results['PredictedMonthlyCharges'].sum()

print("\n===== REVENUE SUMMARY =====")
print(f"Actual Total Revenue: R{total_actual:,.2f}")
print(f"Predicted Total Revenue: R{total_pred:,.2f}")

print(f"\nActual Monthly Revenue: R{monthly_actual:,.2f}")
print(f"Predicted Monthly Revenue: R{monthly_pred:,.2f}")

# ============================================
# 12. GROUP ANALYSIS (NO ML)
# ============================================

# Churn rate by AgeGroup
churn_age = df_original.groupby('AgeGroup')['Churn'].mean()
print("\n===== CHURN RATE BY AGE GROUP =====")
print(churn_age)

# Churn rate by TenureGroup
churn_tenure = df_original.groupby('TenureGroup')['Churn'].mean()
print("\n===== CHURN RATE BY TENURE GROUP =====")
print(churn_tenure)