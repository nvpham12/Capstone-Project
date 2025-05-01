# Import Libraries
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from scipy.stats.mstats import winsorize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

def fit_ols_model(X, y, model_name):
    """Fits an OLS Regression model for the given variables and returns predictions."""
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Get predictions
    y_pred = model.predict(X)
    return model, y_pred


def calculate_vif(X, model_name):
    """Calculates Variance Inflation Factors (VIFs)."""
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


def error_metrics(y, y_pred):
    """Computes error metrics."""
    
    mse = round(mean_squared_error(y, y_pred), 3)
    rmse = round(np.sqrt(mse), 3)
    mae = round(mean_absolute_error(y, y_pred), 3)
    mpe = round(np.mean((y - y_pred) / y) * 100, 3)
    mape = round(np.mean(np.abs((y - y_pred) / y)) * 100, 3)
    r2 = r2_score(y, y_pred)
    return {
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
        "Mean Absolute Error": mae,
        "Mean Percentage Error": mpe,
        "Mean Absolute Percentage Error": mape,
        "R-Squared": r2
    }


def find_resid_sum(y, y_pred):
    """Function to compute the Sum of Residuals."""
   
    rs = np.sum(y - y_pred)
    return rs

    
def find_sae(y, y_pred):
    """Function to compute the Sum of Absolute Errors."""
    
    sae = np.sum(np.abs(y - y_pred))
    return sae


def fit_nn_model(X, y, model_name):
    """Fits a Neural Network model for the given variables and returns predictions and model."""
    
    # Set the input dimension
    input_dim = X.shape[1]

    # Define the neural network with dropout and l2 regularization
    model = Sequential([
        Input(shape=(input_dim,)), # Input layer
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)), # Hidden Layer
        Dropout(0.2), # Dropout to prevent overfitting
        Dense(1, activation='linear', kernel_regularizer=l2(0.01)) # Output Layer
    ])

    # Compile the model and fit with early stopping
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X, y, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2, 
        callbacks=[early_stopping], 
        verbose=0
    )

    # Get predictions
    y_pred = model.predict(X).flatten()
    return model, y_pred, history

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42) 
tf.random.set_seed(42)

st.title("Federal Funds Rate Regression Modeling Dashboard")

with st.expander("Introduction"):
    st.markdown("""
    In this project we will construct models for the Federal Funds Effective Rate. 
    This is a rate determined by the market, similar to stock prices in the stock market. 
    The Federal Funds Rate is a key tool used in monetary policy that influences economic activity. 
    The Federal Reserve sets a Target for the Federal Funds Rate, then performs operations such as trading bonds to adjust the Federal Funds Rate, bringing it closer to the Target. 
    Predicting the Federal Funds Rate benefits educators, economists, investors, financial institutions, and policy planners.

    This project aims to first reproduce regression models predicting the Federal Funds Rate using Taylor’s Rule, a policy guideline by John Taylor from Stanford in 1993 
    and a modification of this used by researcher Alper D. Karakas, the equations of which are derived in Karkas’ (2023) paper, “Reevaluating the Taylor Rule with Machine Learning.”

    We will then attempt to construct other models by adding the Target and Unemployment Rate to the Taylor Model to see if the addition of new features can 
    improve the performance of regression models in predicting the Federal Funds Effective Rate. We choose to build off the Taylor Model as this is the foundational model 
    and Karakas (2023) found little difference in performance between this model and their model.

    We will also check the assumptions of regression for each model. We will also construct simple Neural Network Models for the regression equations.

    Possible stakeholders for this project include policy makers, economists, investors, and lending institutions.
    """)

tabs = st.tabs(["Data Cleaning", "Data Exploration", "Models", "Regression Assumptions", "OLS Model Results", "NN Model Results", "Conclusion", "References"])

# Define a dictionary with datasets and their URLs
datasets = {
    "ffer": "https://raw.githubusercontent.com/nvpham12/Capstone-Project/refs/heads/main/FFER.csv",
    "pgdp": "https://raw.githubusercontent.com/nvpham12/Capstone-Project/refs/heads/main/PGDP.csv",
    "rgdp": "https://raw.githubusercontent.com/nvpham12/Capstone-Project/refs/heads/main/RGDP.csv",
    "cpi": "https://raw.githubusercontent.com/nvpham12/Capstone-Project/refs/heads/main/CPI.csv",
    "fftr_lower": "https://raw.githubusercontent.com/nvpham12/Capstone-Project/refs/heads/main/FFTR_lower.csv",
    "fftr_upper": "https://raw.githubusercontent.com/nvpham12/Capstone-Project/refs/heads/main/FFTR_upper.csv",
    "fftr_old": "https://raw.githubusercontent.com/nvpham12/Capstone-Project/refs/heads/main/FFTR_old.csv",
    "unrate": "https://raw.githubusercontent.com/nvpham12/Capstone-Project/refs/heads/main/UNRATE.csv"
}

# Data Cleaning Tab
with tabs[0]:
    st.header("Data Cleaning")

    # Iterate and load datasets
    for name, url in datasets.items():
        globals()[name] = pd.read_csv(url, parse_dates=["observation_date"])
            
    st.write("All datasets loaded successfully.")

    with st.expander("Data Missing Value and Duplicate Row Check (After Loading)"):
        # Iterate through the dataset names in the dictionary
        for name in datasets.keys():
            # Access the DataFrame using globals()
            dataframe = globals()[name]

            # Find missing values
            missing_val = dataframe.isna().sum()
            total_missing = missing_val.sum()

            # Find duplicates
            duplicates = dataframe[dataframe.duplicated(keep=False)]

            # Print a message indicating if there are any missing values or not
            if total_missing > 0:
                st.write(f"'{name}' has missing values:")
                st.write(missing_val[missing_val > 0])
            else:
                st.write(f"'{name}' has no missing values.")
                
            # Print a message indicating if there are any duplicate values or not
            if not duplicates.empty:
                st.write(f"'{name}' has {len(duplicates)} duplicate rows:")
                st.write(duplicates)
            else:
                st.write(f"'{name}' has no duplicate rows.\n")
            
    # The data for CPI can be used to find inflation rates. 
    # This is done to obtain a seasonally adjusted inflation dataset that isn't available on FRED.
    inflation = pd.DataFrame()
    inflation["observation_date"] = cpi["observation_date"] 
    inflation["Inflation"] = cpi["CPIAUCSL"].pct_change(periods=12) * 100
    inflation.dropna(inplace=True)

    # The Federal Funds Target Rate (FFTR) is set by the Federal Reserve. 
    # The Fed used to set a single value as the target, but they shifted to setting a range.
    # Find the midpoint of the range.

    fftr_midpoint = pd.DataFrame()
    fftr_midpoint["observation_date"] = fftr_upper["observation_date"] 
    fftr_midpoint["Target"] = fftr_upper["DFEDTARU"] - fftr_lower["DFEDTARL"]

    # Combine the midpoint with the old FFTR to get a complete FFTR dataset.
    fftr_old = fftr_old.rename(columns = {"observation_date": "observation_date", "DFEDTAR": "Target"})
    fftr = pd.concat([fftr_old, fftr_midpoint])

    # Resample datasets to daily frequency.
    inflation = inflation.set_index("observation_date").resample("D").ffill().reset_index()
    pgdp = pgdp.set_index("observation_date").resample("D").ffill().reset_index()
    rgdp = rgdp.set_index("observation_date").resample("D").ffill().reset_index()
    unrate = unrate.set_index("observation_date").resample("D").ffill().reset_index()

    # Merge the dataframes
    df = ffer.merge(inflation, on="observation_date", how="outer") \
            .merge(pgdp, on="observation_date", how="outer") \
            .merge(rgdp, on= "observation_date", how="outer") \
            .merge(unrate, on="observation_date", how="outer") \
            .merge(fftr, on="observation_date", how="outer")

    # Set date as an index
    df = df.set_index("observation_date")
    df.index = pd.to_datetime(df.index).date

    # Rename the columns
    df.columns = ["Federal Funds Rate", "Inflation", "Potential GDP", "GDP", "Unemployment", "Target"]

    # Find the inflation gap and output gap and add them to the dataframe
    df["Inflation Gap"] = df["Inflation"] - 2
    df["Output Gap"] = df["GDP"] - df["Potential GDP"]

    # Find Inflation Lag and Output Gap Lag for Karakas Model
    df["Inflation Lag"] = df["Inflation"].shift(1)
    output_gap_lag = df["Output Gap"].shift(1)

    # Create a percentage versions of Output Gap Lag and Inflation Lag for Karakas Model
    df["Output Gap Lag %"] = (output_gap_lag / df["Potential GDP"]) * 100

    # Drop rows with missing values from table
    df.dropna(inplace=True)

    # Drop Potential GDP and GDP columns as they are not needed
    df = df.drop(["Potential GDP", "GDP"], axis=1)

    # Check for duplicates
    st.write("Number of Duplicate Rows in Dataframe (After Merging Datasets):")
    num_duplicates = df.reset_index().duplicated().sum()
    st.write(num_duplicates)

    # Check for missing values
    st.write("Number of Missing Values in Dataframe (After Merging) Datasets:")
    st.write(df.isna().sum().sum())

    # Define the percentage of extreme values to cap
    winsor_limits = (0.05, 0.05)

    # Apply Winsorization to all numeric columns except the dependent variable
    for col in df.columns:
        if col != "Federal Funds Rate":
            lower = np.percentile(df[col], winsor_limits[0] * 100)
            upper = np.percentile(df[col], 100 - winsor_limits[1] * 100)
            df[col] = np.clip(df[col], lower, upper)

    st.write("Outliers have been capped with Winsorization.")
    st.write(f"Winsor Limits: {winsor_limits}")

    # Convert df.dtypes to a DataFrame for better display
    df_dtypes_table = df.dtypes.astype(str).reset_index()
    df_dtypes_table.columns = ["Column Name", "Data Type"]

    # Display the DataFrame
    st.subheader("Data types")
    st.dataframe(df_dtypes_table)
    st.markdown("The data is entirely in float64 with data types consistent across features. This type is perfect for our models and does not require any further action.")

# EDA Tab
with tabs[1]:
    st.header("Exploratory Data Analysis")

    # Print the data table
    st.subheader("Data Table")
    st.dataframe(df)

    # Detailed Information of Features in Data
    with st.expander("Data Feature Information"):
        st.markdown("""
        Federal Funds Rate, Inflation, Unemployment, Target, Inflation Gap, Inflation Lag, and Output Gap Lag % are percentage rates.

        Output Gap is a numerical value in billions of dollars.

        - **Federal Funds Rate**: The interest rate that banks use when lending to each other overnight. This is determined by the market.

        - **Inflation**: The percentage rate at which prices increase.

        - **Unemployment**: The ratio of people without jobs to the total labor force.

        - **Target**: The Federal Funds Rate that the Federal Reserve wants to set as part of economic and monetary policy.

        - **Inflation Gap**: The difference between Inflation and the Inflation Target (the Inflation Target is treated as a constant 2% in this project).

        - **Output Gap**: The difference between Real and Potential Gross Domestic Product (GDP). Measures the difference between how many goods and services a country produces each year vs how much it can produce in theory when using all available resources.

        - **Inflation Lag**: The 1st lagged values of Inflation.

        - **Output Gap Lag %**: The 1st lagged percentage of Output Gap.
        """)

    # Compute Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Plots")
    # Selector for column names
    column = st.selectbox("Select a feature to visualize for the following Line and Distribution Plots:", df.columns)
    with st.expander("Line Plots of Features Over Time"):
        # Plot selected variable over time
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=df.index, y=df[column], ax=ax, color="dodgerblue")
        ax.set_title(f"{column} over time")
        st.pyplot(fig)

        # Notes
        st.markdown("""
            The Federal Funds Rate and Target have been steadily decreasing since the 1980s. 

            Inflation, Output Gap, and Unemployment have roughly remained around a constant level over time, despite having some sharp rises or drops.
            """)

    with st.expander("Distribution Plots"):
        # Plot the distribution of the selected variable
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[column], kde=True, ax=ax, color="dodgerblue")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
        # Notes
        st.markdown("None of the distributions are normal. Some are skewed and others are multimodal.")

    st.subheader("Correlation Heatmap")
    with st.expander("Correlation Heatmap and Analysis"):
        # Construct a Correlation Heatmap
        fig2, ax2 = plt.subplots()
        sns.heatmap(
            df.corr(),
            annot=True,
            cmap="cividis",
            fmt=".2f",
            linewidths=0.5,
        )
        ax2.set_title("Correlation Heatmap")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

        # Notes
        st.markdown("""     
        Inflation, Inflation Gap, and Inflation Lag are perfectly correlated to each other, 
        while Output Gap and Output Gap Lag are very strongly correlated, 
        which is expected since they are the base and derived features. We will not be trying 
        any models that use more than one in each set as predictors at a time, so this will not be an issue.

        Unemployment and Output Gap are strongly correlated, so we will need to watch out for these 
        when checking the Variance Inflation Factors (VIFs) for multicollinearity issues.
     
        Target is highly correlated with our dependent variable, Federal Funds Rate, 
        and we expect it to have the biggest impact on predictive performance of the model. 
        Other features have between 40% and 50% correlation with the dependent variable, which is moderate. 
        Unemployment has a correlation of almost 0 to the dependent variable, 
        which could mean a minimal impact on predictive performance or a non-linear relationship.
        """)

# Model Info Tab
with tabs[2]:
    st.header("Models")
    st.markdown("""
    The variables used for each Regression Model are listed as follows:

    Taylor Rule Model:

        Dependent variable: Federal Funds Rate. 
        Independent variables: Output Gap, Inflation Gap

    Karakas Model:

        Dependent variable: Federal Funds Rate. 
        Independent variables: Output Gap Lag %, Inflation Lag
        
    Target Model:

        Dependent variable: Federal Funds Rate. 
        Independent variables: Output Gap, Inflation Gap, Target
        
    Unemployment Model:

        Dependent variable: Federal Funds Rate. 
        Independent variables: Output Gap, Inflation Gap, Unemployment
        
    Both Model (uses both Unemployment and Target):

        Dependent variable: Federal Funds Rate. 
        Independent variables: Output Gap, Inflation Gap, Target, Unemployment

    The data is first scaled using MinMaxScaler before fitting the models using Ordinary Least Squares (OLS).
    """)

# Model Assumptions Tab
with tabs[3]:
    st.header("Regression Model Assumptions")

    # Apply MinMaxScaler to independent variables
    scaler = MinMaxScaler()
    df_vars = ["Unemployment", "Target", "Inflation Gap", "Output Gap", "Output Gap Lag %", "Inflation Lag"]
    df[df_vars] = scaler.fit_transform(df[df_vars])

    # Set dependent variable
    y = df["Federal Funds Rate"]

    # Define independent variables for each model
    model_features = {
        "Taylor": df[["Output Gap", "Inflation Gap"]],
        "Karakas": df[["Output Gap Lag %", "Inflation Lag"]],
        "Target": df[["Output Gap", "Inflation Gap", "Target"]],
        "Unemployment": df[["Output Gap", "Inflation Gap", "Unemployment"]],
        "Both": df[["Output Gap", "Inflation Gap", "Target", "Unemployment"]],
    }

    # Initialize dictionaries to store results
    ols_fitted_models = {}
    ols_predictions = {}
    ols_vif_results = {}
    ols_error_metrics = {}

    # Ensure session state is initialized for all results
    if "ols_results" not in st.session_state:
        # Initialize dictionaries to store results in session state
        st.session_state["ols_results"] = {
            "fitted_models": {},
            "predictions": {},
            "vif_results": {},
            "error_metrics": {},
            "assumption_tests": {},
        }

        for model_name, X in model_features.items():
            X = sm.add_constant(X)

            # Fit the OLS model
            model, y_pred = fit_ols_model(X, y, model_name)
            st.session_state["ols_results"]["fitted_models"][model_name] = model
            st.session_state["ols_results"]["predictions"][model_name] = y_pred

            # Calculate VIF
            vif_data = calculate_vif(X, model_name)
            st.session_state["ols_results"]["vif_results"][model_name] = vif_data

            # Calculate error metrics
            metrics = error_metrics(y, y_pred)
            st.session_state["ols_results"]["error_metrics"][model_name] = metrics

            # Calculate residuals and assumption tests
            residuals = model.resid
            jb_test = jarque_bera(residuals)
            bp_test = het_breuschpagan(residuals, X)
            dw_stat = sm.stats.durbin_watson(residuals)
            rainbow = linear_rainbow(model)

            st.session_state["ols_results"]["assumption_tests"][model_name] = {
                "Durbin-Watson Test Statistic": f"{dw_stat:.4f}",
                "Jarque-Bera p-value": f"{jb_test[1]:.4f}",
                "Breusch-Pagan p-value": f"{bp_test[1]:.4f}",
                "Rainbow Test p-value": f"{rainbow[1]:.4f}",
            }

    # Combine VIF results into a single DataFrame
    all_vifs = []
    for model_name, vif_df in st.session_state["ols_results"]["vif_results"].items():
        vif_df = vif_df.copy()
        vif_df["Model"] = model_name
        all_vifs.append(vif_df)

    ols_vif_results_df = pd.concat(all_vifs, ignore_index=True).round(4)
    ols_vif_results_df = ols_vif_results_df.pivot(index="Feature", columns="Model", values="VIF").round(4)
    ols_vif_results_df = ols_vif_results_df[["Taylor", "Karakas", "Target", "Unemployment", "Both"]]

    st.write("Test Statistics and P-values for Regression Assumptions")
    # Display assumption tests
    ols_assumption_tests_df = pd.DataFrame(st.session_state["ols_results"]["assumption_tests"]).T.round(4)
    st.dataframe(ols_assumption_tests_df)

    # Assumption Notes
    with st.expander("Regression Assumption Test Statistics and P-values Notes"):
        st.markdown("""
        The assumptions of regression include:

        1. Normality of residuals
        2. Homoscedasticity (constant variance of residuals)
        3. Independence (autocorrelation)
        4. Linearity
        5. No Multicollinearity

        These assumptions can be tested using the Jarque-Bera test (Normality), Breusch-Pagan test (Homoscedasticity), Durbin-Watson test (Independence), and Rainbow test (Linearity).
        We perform a simple hypothesis test for these where the hypotheses are as follows:
        
        Null Hypothesis, H0: The model does not violate the regression assumption

        Alternative Hypothesis, H1: The model does violate the regression assumption

        Using the 95% confidence level (significance level 0.05), the Jarque-Bera, Breusch-Pagan, and Rainbow test statistics all have p-values of around 0, which is less than the significance level. 
        Therefore, we would reject the null hypothesis, H0, in favor of the alternative and conclude that each model violates the assumptions for these tests.

        For the Durbin-Watson test, any test statistics less than 1 or greater than 3 indicate strong autocorrelation and would violate the regression assumption of independence.
        Since the Durbin Watson test statistics for every model lies between 0 and 0.5, we would reject the null hypothesis and conclude the assumption of independence has been violated for each model. 

        All models violate the first 4 regression Assumptions. When regression assumptions are violated, any metrics become biased and unreliable.             
        """)

    # Display VIF table
    st.write("Variance Inflation Factors")
    st.dataframe(ols_vif_results_df)

    # VIF Notes
    with st.expander("VIF Notes"):
        st.markdown("""
        Variance Inflation Factors (VIFs) are used to test for multicollinearity. Any null values here are due to the feature not being included in the model.
        The threshold is that VIFs below 5 have no issues. 
        From the VIFs, we do not have serious problems with multicollinearity except for the Output Gap in the Both Model. 
        The best practice would to remove that from the model entirely, but since all other regression assumptions have been violated we will leave it and run the regression anyway. 
        Unemployment, the other variable of interest from our correlation matrix analysis, has acceptable multicollinearity levels.
        """)

# Model Performance Tab
with tabs[4]:
    st.header("Model Performance")

    # Print Table of Error metrics for all OLS models
    st.subheader("OLS Model Metrics")
    ols_error_metrics_df = pd.DataFrame(st.session_state["ols_results"]["error_metrics"]).T.round(4)
    st.dataframe(ols_error_metrics_df)
    
    with st.expander("OLS Model Error Metrics Notes"):
        # Notes
        st.markdown("""
        The error metrics show the Taylor Model having better performance over the Karakas Model. 
        Karakas (2023) claimed that their model had more accurate predictions, but not by much. 

        We, on the other hand, found that the Taylor model has better predictions. 
        Note that we have a different date range than Karakas, and this could mean that Karakas' model better predicts older data, but performs poorly on more recent data. 
        Overall, their model does not perform as well as the Taylor Model, having larger average errors and percentage errors. 
        The difference, however, is small, which is consistent with Karakas' findings.

        All models have negative Mean Percentage Errors, and thus they all underpredict the Federal Funds Rate.
        The percentage error metrics seem to be very high. This is likely because due to our data having small values. 
        Average error metrics would be better for analysis. 
        
        In terms of performance, the models from worst to best are:
        
        Karakas < Taylor < Unemployment < Target < Both

        Models with lower error metrics and higher R-squared are considered better performing than others.

        Adding Target or Unemployment alone to the Taylor Model increases performance, but there are marginal differences in performance when adding Unemployment alongside the Target. 
        This suggests that Unemployment contributes little to the model, consistent with the findings from the correlation heatmap. 
        
        Note that metrics are likely biased and unreliable since regression assumptions have been violated.
        """)

    # Initialize a list to store model statistics
    model_statistics = []

    # Loop through the fitted models to extract key statistics
    for model_name, model in  st.session_state["ols_results"]["fitted_models"].items():
        # Extract key values
        results = {
            "model": model_name,
            "adj_r_squared": round(model.rsquared_adj, 3),
            "aic": round(model.aic, 3),
            "bic": round(model.bic, 3),
            "f_stat": round(model.fvalue, 3),
            "f_p_value": round(model.f_pvalue, 3),
            "t_p_values": model.pvalues.round(3).tolist(),
        }
        # Append the results for each model
        model_statistics.append(results)

    # Convert results to a DataFrame
    statistics = pd.DataFrame(model_statistics)

    # Print table of Adjusted R-squared, Information Criteria, and P-values
    st.subheader("Additional OLS Model Statistics")
    st.dataframe(statistics)
    with st.expander("Additional OLS Model Statistics Notes"):
        # Notes
        st.markdown("""
        A model with lower values for AIC and BIC is better than a model with higher values for them. 
        The AIC and BIC show a similar pattern to our findings from the error metrics and R-squared, with the models listed from worst performing to best performing being:

        Karakas < Taylor < Unemployment < Target < Both.

        The p-values for the f-statistic and t-statistic suggests the models and coefficients are almost all statistically significant from a simple hypothesis test where the null hypothesis, 
        H0, is that the models/coefficients are not significant. The exception is the constant from the Taylor Model which would not be statistically significant.

        Note that metrics are likely biased and unreliable since regression assumptions have been violated.
        """)

    st.subheader("Validation of Models from Karakas' Paper")
    with st.expander("Taylor and Karakas Model Metrics Comparison"):
        # Define models, initialize dictionary to store metrics and calculate metrics
        taylor_and_karakas = ["Taylor", "Karakas"]
        metrics = []
        
        for model in taylor_and_karakas:
            rs = find_resid_sum(y, st.session_state["ols_results"]["predictions"][model])
            sae = find_sae(y, st.session_state["ols_results"]["predictions"][model])
            metrics.append({"Model": model, "Sum of Residuals": rs, "Sum of Absolute Errors": round(sae, 4)})
            
        # Convert metrics to a DataFrame
        metrics_df = pd.DataFrame(metrics)
            
        # Display metrics as a table
        st.dataframe(metrics_df)
    
        # Notes
        st.markdown("""
        We got much smaller Sum of Residuals than Karakas did and obtained larger Sum of Absolute Errors for both models. 
        Additionally, we received lower values for each in the Taylor Model than the Karakas Model.  

        The metrics for each model are very close to each other, 
        which implies that there isn't much of a difference in performance between the two models and 
        is consistent with Karakas' findings.
        """)

    with st.expander("Taylor vs Karakas Predictions Plot"):
        fig, ax = plt.subplots()
        ax.plot(df.index, y, label="True Federal Funds Rate", color="dodgerblue")
        ax.plot(df.index, st.session_state["ols_results"]["predictions"]["Taylor"], label="Taylor Predictions", color="darkorange", linestyle="--")
        ax.plot(df.index, st.session_state["ols_results"]["predictions"]["Karakas"], label="Karakas Predictions", color="purple", linestyle=":")
        ax.set_title("Taylor vs Karakas Predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Federal Funds Rate")
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        We capture similar patterns such as the Karakas Model underpredicting more than Taylor Model between 1997 to 2001 and mostly between 2010 and 2015, 
        where there are 2 cross overs as they switch between overpredicting or underpredicting each other.
        """)

    st.subheader("Model Predictions Plot")
    # Selector for OLS Model
    ols_model = st.selectbox("Select an OLS model to visualize:", list(st.session_state["ols_results"]["predictions"].keys()))

    # Plot Model Predictions vs Actual
    fig, ax = plt.subplots()
    ax.plot(df.index, y, label="True Federal Funds Rate", color="dodgerblue")
    ax.plot(df.index, st.session_state["ols_results"]["predictions"][ols_model], label=f"{ols_model} Predictions", color="darkorange", linestyle="--")
    ax.set_title(f"{ols_model} vs Federal Funds Rate")
    ax.set_xlabel("Date")
    ax.set_ylabel("Federal Funds Rate")
    ax.legend()
    st.pyplot(fig)

    if ols_model == "Taylor":
        # Notes
        st.markdown("""
        Consistent with Karakas' plots, we see our Taylor Model underpredicting in year 1990, briefly overpredicting between 1990 and 1995, before underpredicting until a bit past 2000. 
        Around year 2002, the model begins overpredicting for the rest of the years, except between 2005 and 2010 where it underpredicts where the actual values form a peak and around 2008 or 2009.
        The plots may not be an exact match but they look similar enough to the ones shared in Karakas' paper. 
        """)
    st.markdown("""
    The plot of predictions remains consistent with our findings from the error metrics. 
    Adding Target and Unemployment improve model predictions, but they tend to more closely follow earlier years of the data.
    """)
# Neural Network Models Tab
with tabs[5]:
    st.header("Neural Network Models")

    # Initialize dictionaries to store results
    nn_fitted_models = {}
    nn_predictions = {}
    nn_error_metrics = {}
    nn_history = {}

    # Ensure session state is initialized
    if "nn_predictions" not in st.session_state:
        # Train and save the models only once
        st.session_state["nn_fitted_models"] = {}
        st.session_state["nn_predictions"] = {}
        st.session_state["nn_error_metrics"] = {}
        st.session_state["nn_history"] = {}

        for model_name, X in model_features.items():
            # Fit the model
            model, y_pred, history = fit_nn_model(X, y, model_name)
            
            # Store predictions and metrics in session state
            st.session_state["nn_fitted_models"] = model
            st.session_state["nn_predictions"][model_name] = y_pred
            st.session_state["nn_error_metrics"][model_name] = error_metrics(y, y_pred)
            st.session_state["nn_history"] = history

    # Error metrics for all NN models
    nn_error_metrics_df = pd.DataFrame(st.session_state["nn_error_metrics"]).T.round(4)
    st.dataframe(nn_error_metrics_df)

    with st.expander("Neural Network Model Error Metrics Notes"):
        # Notes
        st.markdown("""
        Our models have returned worse results for each metric compared to the OLS models. The regression assumptions being violated may have inflated the OLS model metrics. 
        Our model performance ranking differs slightly here.
        
        We now have from worst performing to best performing: 

        Karakas < Taylor < Unemployment < Both < Target. 

        Like with the OLS models, we find that there is little difference in performance between the Karakas and Taylor Models.
        
        All models have negative Mean Percentage Errors, meaning that all of the models underpredict the Federal Funds Rate.

        The inclusion of both the Target and Unemployment into the Taylor Model raised the R-squared from around 0.24 to around 0.89, which is very strong performance for a model that uses economic data. 
        However, there is not much of a difference between the performance of the Target Model and the Both Model in average errors or R-squared. 
        On the other hand, the MPE and MAPE have differences of around 27% to 40%. These values are likely high due to our data having small values. 
        Average error metrics would be better for analysis. 

        Adding Unemployment to the Taylor Model alone reduces errors and explains more variance but increases percentage errors. Unemployment contributes little when added with the Target. 
        """)
    
    st.subheader("Neural Network Prediction Plots")
    # Selector for model names
    nn_model = st.selectbox("Select a NN model to visualize:", list(st.session_state["nn_predictions"].keys()))

    # Plot the Model Predictions vs Federal Funds Rate
    fig3, ax3 = plt.subplots()
    ax3.plot(df.index, y, label="True Federal Funds Rate", color="dodgerblue")
    ax3.plot(df.index, st.session_state["nn_predictions"][nn_model], label=f"{nn_model} Predictions", color="darkorange", linestyle="--")
    ax3.set_title(f"{nn_model} vs Federal Funds Rate")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Federal Funds Rate")
    ax3.legend()
    st.pyplot(fig3)

    # Plot Notes
    st.markdown("""
    Each model tends to underpredict the Federal Funds Rate in the first half of the date range (around 1980 to 2000). Afterwards, the models tend to overpredict.

    The Taylor, Karakas, and Unemployment Models tend to predict Federal Funds Rates of around 4. The Target and Both Models follow the true Federal Funds Rates more closely, having smaller gaps between them and their predictions. 
    """)

# Conclusion Tab
with tabs[6]:
    st.header("Conclusion")
    st.markdown("""
    Although the results may differ due to different date ranges in our data, our models captured similar predictive patterns to those presented in Karakas' paper 
    in our examination of the plots as well as the Sum of Residuals and Sum of Absolute Errors. 

    Karakas (2023) noted that the Taylor Model (and their transformation of it) did not predict the Federal Funds Rate well and 
    mentioned their model having predictions closer to the actual Federal Funds Rate than the Taylor Model, but not by much. 
    However, our models' metrics show that it is the Taylor Model, rather than Karakas' Model that has smaller average errors and percentage errors. 
    We suspect this may be due to how the data we use contains more recent data. Also, our plot of the Taylor Model predictions does not have any predictions below 0 unlike Karakas' plot. 
    This may be because Karakas did not handle outliers well if at all.

    We noticed that Karakas did not mention regression assumptions in their paper when discussing their OLS models, so we decided to check them as part of the validation process.
    We found that almost all classical regression assumptions were violated, with the exception being multicollinearity. 

    Karakas later created neural network models using Taylor's Rule as well, getting better predictions than their OLS models. 
    From our findings, the OLS models had inflated metrics due to regression assumption violations.
    Our neural network models, on the other hand, had lower performance metrics than our OLS models. 
    Since Karakas got better results from their neural network model than their OLS model, which should also have inflated metrics, 
    their neural network is likely to have issues with overfitting given that they did not mention usage of techniques such as regularization, dropout, or early stopping like we had for our models.

    This casts serious doubts about the credibility, rigor, and professionalism of their work.

    We believe that non-linear models are better suited for predicting the Federal Funds Rate since no feature we used had a linear relationship with the Federal Funds Rate.  
    
    It is worth noting that the inclusion of Target and Unemployment to the Taylor Model, individually, did improve performance metrics in both the OLS and Neural Network Models. 
    However, the Target has a much larger effect on model performance than Unemployment and Unemployment has little effect on performance when it is alongside the Target.

    While performance metrics look strong for the Target Model, there is room for improvement. 
    Some next steps would be to explore interaction terms with Unemployment or taking lags of Unemployment to avoid multicollinearity issues with Output Gap. 
    The addition of other features such as Global Economic Growth, Prices, Interest rates, Foreign Currency Exchange Rates, Consumer Confidence, and Business Confidence are also worth exploring.
    """)

# Data Sources and References
with tabs[7]:
    st.header("References")
    st.markdown("""
    Data obtained from:

    Board of Governors of the Federal Reserve System (US). (2025). Federal Funds Effective Rate [DFF]. Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/DFF

    Board of Governors of the Federal Reserve System (US). (2025). Federal Funds Target Rate (DISCONTINUED) [DFEDTAR]. Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/DFEDTAR

    Board of Governors of the Federal Reserve System (US). (2025). Federal Funds Target Range - Lower Limit [DFEDTARL]. Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/DFEDTARL

    Board of Governors of the Federal Reserve System (US). (2025). Federal Funds Target Range - Upper Limit [DFEDTARU]. Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/DFEDTARU

    U.S. Bureau of Economic Analysis. (2025). Real Gross Domestic Product [GDPC1]. Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/GDPC1

    U.S. Bureau of Labor Statistics. (2025). Consumer Price Index for all urban consumers: All items in U.S. city average [CPIAUCSL]. Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/CPIAUCSL

    U.S. Bureau of Labor Statistics. (2025). Unemployment Rate [UNRATE]. Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/UNRATE

    U.S. Congressional Budget Office. (2025). Real Potential Gross Domestic Product [GDPPOT]. Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/GDPPOT

    Karakas' Paper:

    Karakas, A. D. (2023). Reevaluating the Taylor Rule with Machine Learning. ArXiv.org. https://arxiv.org/abs/2302.08323
    """)

gc.collect()  # Free up memory after everything runs
