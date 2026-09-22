import datetime

import pytz as pytz
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from model.helpers import train, predict
from model.preprocessors import process_inputs, process_targets

if __name__ == "__main__":
    # Download price histories from Yahoo Finance
    spy = yf.Ticker("SPY")
    price_series = spy.history(period='max')['Close'].dropna()

    perf_series = price_series.pct_change().dropna()

    x_df = process_inputs(perf_series, window_length=10)
    y_series = process_targets(perf_series)

    # Only keep rows in which we have both inputs and data.
    common_index = x_df.index.intersection(y_series.index)
    x_df, y_series = x_df.loc[common_index], y_series.loc[common_index]

    # Isolate training data
    training_cutoff = datetime.datetime(2020, 1, 1, tzinfo=pytz.timezone('America/New_York'))
    training_x_series = x_df.loc[x_df.index < training_cutoff]
    training_y_series = y_series.loc[y_series.index < training_cutoff]

    trained_model = train(training_x_series, training_y_series)

    # Isolate test data
    test_x_series = x_df.loc[x_df.index >= training_cutoff]
    actual_series = y_series.loc[y_series.index >= training_cutoff]

    forecast_series = predict(trained_model, test_x_series)
    results_df = forecast_series.to_frame('Forecast').join(actual_series.to_frame('Actual')).dropna()

    # Evaluate forecasts
    results_df.plot.scatter(x='Actual', y='Forecast')
    plt.show()

    r2 = r2_score(results_df['Actual'], results_df['Forecast'])
    mae = mean_absolute_error(results_df['Actual'], results_df['Forecast'])

    # Directional accuracy: percentage of forecasts with the correct return sign.
    correct_direction = (
        (results_df['Forecast'] > 0) == (results_df['Actual'] > 0)
    )
    directional_accuracy = correct_direction.mean()

    # "Win rate" for long signals: when the model forecasts a positive return,
    # how often is the realized two-trading-day-ahead return also positive?
    positive_signals = results_df[results_df['Forecast'] > 0]
    positive_signal_win_rate = (
        (positive_signals['Actual'] > 0).mean()
        if len(positive_signals) > 0
        else float('nan')
    )

    # Equivalent hit rate for negative forecasts.
    negative_signals = results_df[results_df['Forecast'] < 0]
    negative_signal_win_rate = (
        (negative_signals['Actual'] < 0).mean()
        if len(negative_signals) > 0
        else float('nan')
    )

    print(f"R Squared: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")
    print(
        f"Positive-Signal Win Rate: {positive_signal_win_rate:.2%} "
        f"({len(positive_signals)} signals)"
    )
    print(
        f"Negative-Signal Win Rate: {negative_signal_win_rate:.2%} "
        f"({len(negative_signals)} signals)"
    )
    print(f"Out-of-Sample Observations: {len(results_df)}")
