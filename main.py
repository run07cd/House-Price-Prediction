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

    # The dataset's target is expressed in units of $100,000.
    print("Mean Absolute Error: $" + format(mae * 100000, ",.0f"))
    print("Root Mean Squared Error: $" + format(rmse * 100000, ",.0f"))
    print("R2 score:", round(r2, 3))

    sample_prediction = model.predict([X_test[0]])[0]
    print("Example predicted median house value: $" + format(sample_prediction * 100000, ",.0f"))

    print("\nMost influential features in this fitted model:")
    ranked_features = sorted(
        zip(housing.feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    for name, importance in ranked_features:
        print(f"{name}: {importance:.3f}")

    print("\nThese estimates use the California Housing dataset and are not current property valuations.")


if __name__ == "__main__":
    main()
