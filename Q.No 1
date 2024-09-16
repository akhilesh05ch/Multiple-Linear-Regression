import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv('/content/50_Startups.csv')
X = data[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = data['Profit']
X = pd.get_dummies(X, columns=['State'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_single = X_train[['R&D Spend']]
X_test_single = X_test[['R&D Spend']]
model_single = LinearRegression()
model_single.fit(X_train_single, y_train)
y_pred_single = model_single.predict(X_test_single)
r2_single = r2_score(y_test, y_pred_single)
print(f"Single Linear Regression R-squared: {r2_single}")
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)
y_pred_multiple = model_multiple.predict(X_test)
r2_multiple = r2_score(y_test, y_pred_multiple)
print(f"Multiple Linear Regression R-squared: {r2_multiple}")
