from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    housing = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data,
        housing.target,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", round(mae, 3))
    print("Root Mean Squared Error:", round(rmse, 3))
    print("R2 score:", round(r2, 3))
    print("Example predicted median house value:", round(model.predict([X_test[0]])[0], 3))


if __name__ == "__main__":
    main()
