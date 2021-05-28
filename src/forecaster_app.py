import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

from datetime import timedelta
from fbprophet import Prophet
from fbprophet.make_holidays import make_holidays_df
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.utilities import regressor_coefficients
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from app_conf import _max_width_


# Settings
_max_width_(970)


st.header("A quick time series (prophet) modelling and visualization app")
st.text("This app uses counting points for bicycles in different locations of Berlin")
st.markdown ("https://www.berlin.de/sen/uvk/verkehr/verkehrsplanung/radverkehr/weitere-radinfrastruktur/zaehlstellen-und-fahrradbarometer")


# data
@st.cache()
def get_data(filename="ts_bicycle_and_weather_berlin_day.csv.gz"):
    return pd.read_csv(
        filename, sep="|", index_col=None, encoding="utf8", compression="gzip"
    )

@st.cache()
def get_locations(filename="ts_bicycle_and_weather_berlin_day.csv.gz"):
    return (
        pd.read_csv(
            filename,
            sep="|",
            index_col=None,
            encoding="utf8",
            compression="gzip",
            usecols=["place_description"],
        )["place_description"]
        .unique()
        .tolist()
    )


# selection
st.sidebar.markdown("# Forecast model parameter settings")
st.sidebar.markdown("### Select data and params for the model")

with st.form(key="form"):
    sel_data_granularity = st.sidebar.selectbox(
        "data_granularity",
        ["daily", "hourly"],
        index=0,
    )

    possible_locations = get_locations(
        filename="ts_bicycle_and_weather_berlin_day.csv.gz"
    )
    possible_locations_dict = {
        v: k for k, v in dict(enumerate(possible_locations)).items()
    }

    if sel_data_granularity == "hourly":
        weather_columns = ["temp", "humidity", "cloudcover", "windspeed"]
        filename = "ts_bicycle_and_weather_berlin_hour.csv.gz"

    if sel_data_granularity == "daily":
        weather_columns = ["tempmin", "tempmax", "humidity", "cloudcover", "windspeed"]
        filename = "ts_bicycle_and_weather_berlin_day.csv.gz"

    dfm = get_data(filename=filename)

    sel_location = st.sidebar.selectbox(
        "location",
        possible_locations,
        index=possible_locations_dict["Jannowitzbrücke Süd"],
    )
    dfm_sel = dfm[dfm.place_description == sel_location]
    # Use data from first day > 0
    min_day = min(dfm_sel[dfm_sel.cnt > 0]["dt"])
    dfm_sel = dfm_sel.query("dt >= @min_day")

    sel_seasonality_mode = st.sidebar.selectbox(
        "seasonality_mode",
        ["additive", "multiplicative"],
    )

    sel_yearly_seasonality = st.sidebar.select_slider(
        "yearly_seasonality", list(range(4, 12 + 1)), value=4
    )
    sel_changepoint_prior_scale = st.sidebar.select_slider(
        "changepoint_prior_scale", [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10], value=0.1
    )
    sel_seasonality_prior_scale = st.sidebar.select_slider(
        "seasonality_prior_scale", [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10], value=0.1
    )
    submit_button = st.form_submit_button(
        label="Press buttton - after modified params)"
    )

    if submit_button:

        # Model
        def train_model(
            dfm_sel,
            weather_columns,
            seasonality_mode,
            yearly_seasonality,
            changepoint_prior_scale,
            seasonality_prior_scale,
            forecast_period_days=2,
        ):
            df_model = dfm_sel.reset_index(drop=True)[
                ["dt", "cnt"] + weather_columns
            ].rename({"dt": "ds", "cnt": "y"}, axis=1)
            df_model["ds"] = pd.to_datetime(df_model["ds"])
            train = df_model[df_model["ds"] < df_model["ds"].max() - timedelta(days=2)]

            holidays = make_holidays_df(
                year_list=[year for year in range(2016, 2020)], country="DE", state="BL"
            )

            model = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=yearly_seasonality,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays=holidays,
                uncertainty_samples=False,
            )

            for c in weather_columns:
                model.add_regressor(c)
            model.fit(train)

            # Prepare future
            future = model.make_future_dataframe(periods=forecast_period_days)
            for c in weather_columns:
                future[c] = df_model[c]
            forecast = model.predict(future)
            return model, forecast

        model, forecast = train_model(
            dfm_sel,
            weather_columns=weather_columns,
            seasonality_mode=sel_seasonality_mode,
            yearly_seasonality=sel_yearly_seasonality,
            changepoint_prior_scale=sel_changepoint_prior_scale,
            seasonality_prior_scale=sel_seasonality_prior_scale,
            forecast_period_days=2,
        )

        # Main plot
        font = {"size": 14}
        matplotlib.rc("font", **font)

        st.markdown(f"## Forecast model - tracked bicycles in Berlin - {sel_location}")

        st.markdown(f"### Scatterpplot - {sel_location}")
        fig = model.plot(forecast, figsize=(14, 8))
        add_changepoints_to_plot(fig.gca(), model, forecast)
        plt.xlabel("Date")
        plt.ylabel("Tracked bicycles")

        #  main page
        st.pyplot(fig=fig)

        # Component plot
        # cols = st.beta_columns(2)
        st.markdown(f"### Component plot - {sel_location}")
        font = {"size": 11}
        matplotlib.rc("font", **font)
        fig2 = model.plot_components(forecast)
        st.pyplot(fig=fig2)

        def holiday_effects(model, forecast):
            first_non_zero = lambda f, h: f[f[h] != 0][h].values[0]
            return pd.DataFrame(
                {
                    "holiday": model.train_holiday_names,
                    "effect": [
                        first_non_zero(forecast, holiday)
                        for holiday in model.train_holiday_names
                    ],
                }
            )

        st.markdown(f"### Regressor effects - {sel_location}")
        st.dataframe(regressor_coefficients(model))

        if sel_data_granularity=='daily':
            st.markdown(f"### Holiday effects - {sel_location}")
            st.dataframe(holiday_effects(model, forecast))

            st.markdown(f"### Cross validation - {sel_location}")
            df_cv = cross_validation(
                model,
                horizon="30 days",
                period="30 days",
                initial=f"1000 days",
                parallel="processes",
            )
            df_p = performance_metrics(df_cv)
            df_cv["ds"] = df_cv["ds"].dt.date
            df_cv["cutoff"] = df_cv["cutoff"].dt.date
            st.dataframe(df_cv)

            st.markdown(f"### Performance metrics - {sel_location}")
            df_p["horizon"] = df_p["horizon"].dt.days
            df_p = df_p.rename(columns={"horizon": "horizon_in_days"})
            st.dataframe(df_p)
