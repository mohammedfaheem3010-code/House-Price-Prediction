from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from src.feature_engineering import create_features
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,GradientBoostingRegressor)
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
from sklearn.model_selection import (GridSearchCV,cross_val_score,learning_curve)



print("=" * 60)
print("HOUSE PRICE PREDICTION PROJECT")
print("=" * 60)

housing = fetch_california_housing(as_frame=True)

df = housing.frame

df.to_csv("data/housing.csv", index=False)

print("\nDataset Loaded Successfully!")

print("\nShape of Dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\n" + "=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

print("\nLast 5 Rows:")
print(df.tail())

print("\nRandom Sample:")
print(df.sample(5, random_state=42))

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


print("EXPLORATORY DATA ANALYSIS")
plt.style.use("ggplot")
#1
plt.figure(figsize=(8,5))
sns.histplot(df["MedHouseVal"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Median House Value")
plt.ylabel("Count")
plt.savefig("graphs/01_house_price_distribution.png")
plt.show()

#2
plt.figure(figsize=(8,5))
sns.histplot(df["MedInc"], bins=30, color="green", kde=True)
plt.title("Median Income Distribution")
plt.xlabel("Median Income")
plt.ylabel("Count")
plt.savefig("graphs/02_income_distribution.png")
plt.show()

#3
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("graphs/03_correlation_heatmap.png")
plt.show()

#4
plt.figure(figsize=(8,5))
sns.scatterplot(x="HouseAge",y="MedHouseVal",data=df)
plt.title("House Age vs House Price")
plt.savefig("graphs/04_houseage_vs_price.png")
plt.show()

#5
plt.figure(figsize=(8,5))
sns.scatterplot(x="MedInc",y="MedHouseVal",data=df,color="red")
plt.title("Income vs House Price")
plt.savefig("graphs/05_income_vs_price.png")
plt.show()

#6
plt.figure(figsize=(10,5))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplot for Detecting Outliers")
plt.savefig("graphs/06_boxplot.png")
plt.show()

#7
sns.pairplot(df[["MedInc","HouseAge","AveRooms","Population","MedHouseVal"]])
plt.savefig("graphs/07_pairplot.png")
plt.show()

#8
plt.figure(figsize=(8,6))
sns.scatterplot(x="Longitude",y="Latitude",hue="MedHouseVal",data=df,palette="viridis")
plt.title("California House Locations")
plt.savefig("graphs/08_map.png")
plt.show()

#9
plt.figure(figsize=(8,5))
sns.histplot(df["AveRooms"], bins=30, color="orange")
plt.title("Average Rooms Distribution")
plt.savefig("graphs/09_rooms.png")
plt.show()

#10
plt.figure(figsize=(8,5))
sns.histplot(df["Population"], bins=40)
plt.title("Population Distribution")
plt.savefig("graphs/10_population.png")
plt.show()

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
print("\nFeatures Shape:")
print(X.shape)
print("\nTarget Shape:")
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTrain/Test Split Completed Successfully!")
print("\nTraining Data Shape")
print(X_train.shape)
print("\nTesting Data Shape")
print(X_test.shape)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeature Scaling Completed!")
print("\nScaled Training Shape:")
print(X_train_scaled.shape)
print("\nScaled Testing Shape:")
print(X_test_scaled.shape)
joblib.dump(scaler, "models/scaler.pkl")
print("\nScaler Saved Successfully!")

df = create_features(df)
print(df.head())
print("\nNew Features Added:")
print(df.columns)
correlation = df.corr(numeric_only=True)
print("\nCorrelation with House Price")
print(correlation["MedHouseVal"].sort_values(ascending=False))
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.savefig("graphs/11_feature_correlation.png")
plt.close()
print("Feature Correlation Graph Saved")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
print("\nSelected Features")
print(X.columns)
print("\nDataset Shape")
print(df.shape)

#LINEAR REGRESSION
lr_model=LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("\nModel Training Completed!")
y_pred = lr_model.predict(X_test_scaled)
print("\nPrediction Completed!")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print("\nModel Performance")
print("-"*40)
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")
joblib.dump(lr_model,"models/house_price_model.pkl")
print("\nModel Saved Successfully!")
plt.figure(figsize=(8,6))
plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)
plt.savefig("graphs/12_actual_vs_predicted.png")
plt.close()
print("Actual vs Predicted Graph Saved")
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)
plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("graphs/13_residual_plot.png")
plt.close()
print("Residual Plot Saved")

#ALGORITHMS
print("MODEL COMPARISON")
models = {
    "Linear Regression": lr_model,
    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}
results = []
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    if model_name != "Linear Regression":
        model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    results.append([
        model_name,
        mae,
        mse,
        rmse,
        r2
    ])
    print(f"{model_name} Completed")
    results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ]
)
print("\n")
print("MODEL COMPARISON")
print(results_df)
results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)
print("\n")
print("BEST MODEL")
print(results_df.head(1))
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)
print("\nComparison Report Saved")
plt.figure(figsize=(10,6))
sns.barplot(
    x="Model",
    y="R2 Score",
    data=results_df
)
plt.title("Model Comparison using R² Score")
plt.xticks(rotation=20)
plt.savefig("graphs/14_model_comparison.png")
plt.close()
print("Comparison Graph Saved")



# HYPERPARAMETER TUNING
param_grid = {
    "n_estimators":[100,200],
    "max_depth":[10,20,None],
    "min_samples_split":[2,5],
    "min_samples_leaf":[1,2]
}
rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train_scaled,y_train)
print("\nGrid Search Completed")
print("\nBest Parameters")
print(grid.best_params_)
print("\nBest Cross Validation Score")
print(grid.best_score_)
best_model = grid.best_estimator_
print("\nBest Model")
print(best_model)
best_prediction=best_model.predict(X_test_scaled)
best_mae=mean_absolute_error(y_test,best_prediction)
best_mse=mean_squared_error(y_test,best_prediction)
best_rmse=best_mse ** 0.5
best_r2=r2_score(y_test,best_prediction)
print("\nBest Model Performance")
print(f"MAE:{best_mae:.4f}")
print(f"MSE:{best_mse:.4f}")
print(f"RMSE:{best_rmse:.4f}")
print(f"R2:{best_r2:.4f}")
accuracy = best_r2 * 100
print(f"\nModel Accuracy (R² Percentage): {accuracy:.2f}%")
print("\nBest Model Performance")
print(f"MAE:{best_mae:.4f}")
print(f"MSE:{best_mse:.4f}")
print(f"RMSE:{best_rmse:.4f}")
print(f"R2:{best_r2:.4f}")
accuracy = best_r2 * 100
print(f"\nModel Accuracy (R² Percentage): {accuracy:.2f}%")
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)
results_df["Accuracy (%)"]=results_df["R2 Score"]*100
print(results_df)
scores=cross_val_score(best_model,X_train_scaled,y_train,cv=5,scoring="r2")
print("\nCross Validation Scores")
print(scores)
print("\nAverage CV Score")
print(scores.mean())
joblib.dump(best_model,"models/best_house_price_model.pkl")
print("\nBest Model Saved Successfully")
train_sizes,train_scores,test_scores = learning_curve(
    best_model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1,1.0,5)
)
train_mean=train_scores.mean(axis=1)
test_mean=test_scores.mean(axis=1)
plt.figure(figsize=(8,5))
plt.plot(train_sizes,train_mean,label="Training Score")
plt.plot(train_sizes,test_mean,label="Validation Score")
plt.xlabel("Training Samples")
plt.ylabel("R2 Score")
plt.title("Learning Curve")
plt.legend()
plt.savefig("graphs/15_learning_curve.png")
plt.close()
print("Learning Curve Saved")
residual=y_test - best_prediction
plt.figure(figsize=(8,5))
plt.scatter(best_prediction,residual)
plt.axhline(y=0,color="red",linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.savefig("graphs/16_residual_plot.png")
plt.close()
print("Residual Plot Saved")
plt.figure(figsize=(8,5))
sns.histplot(residual,bins=30,kde=True)
plt.title("Prediction Error Distribution")
plt.savefig("graphs/17_prediction_error_distribution.png")
plt.close()
print("Prediction Error Distribution Saved")
joblib.dump(lr_model, "models/house_price_model.pkl")
joblib.dump(best_model, "models/best_house_price_model.pkl")