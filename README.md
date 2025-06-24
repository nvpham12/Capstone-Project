# Introduction
The Federal Funds Effective Rate is a rate determined by the market, similar to stock prices in the stock market. The Federal Funds Rate is a key tool used in monetary policy that influences economic activity. The Federal Reserve sets a Target for the Federal Funds Rate, then performs operations such as trading bonds to adjust the Federal Funds Rate, bringing it closer to the Target. Predicting the Federal Funds Rate benefits educators, economists, investors, financial institutions, and policy planners.

# Project Overview
This project aims to first reproduce regression models predicting the Federal Funds Rate using Taylor’s Rule, a policy guideline by John Taylor from Stanford in 1993 and a modification of this used by researcher Alper D. Karakas, the equations of which are derived in Karkas’ (2023) paper, “Reevaluating the Taylor Rule with Machine Learning.”

Other models will be constructed by adding the Target and Unemployment Rate to the Taylor Model to see if the addition of new features can improve the performance of regression models in predicting the Federal Funds Effective Rate. The other models are expansions of the Taylor Model as this is the foundational model for Federal Funds Rate prediction and Karakas (2023) found little difference in performance between this model and their model.

The models will be fitted using Ordinary Least Squares (OLS) and Neural Networks (NNs).

For more theoretical details please refer to the documentation. To view the Streamlit web dashboard please visit: https://ffr-modeling-5n85wvsg3gdnvd4z4yz3kh.streamlit.app/

# Data
The data was obtained from the Federal Reserve Economic Data (FRED) database. The datasets downloaded and used for this project are
- Federal Funds Effective Rate [DFF]
- Federal Funds Target Rate (DISCONTINUED) [DFEDTAR]
- Federal Funds Target Range - Lower Limit [DFEDTARL]
- Federal Funds Target Range - Upper Limit [DFEDTARU]
- Real Gross Domestic Product [GDPC1]
- Consumer Price Index for all urban consumers: All items in U.S. city average [CPIAUCSL]
- Unemployment Rate [UNRATE]
- Real Potential Gross Domestic Product [GDPPOT]

# Data Wrangling
- All datasets were checked for missing values or duplicates before further procedure.
- The Lower Limit and Upper Limit Datasets were combined to create a new dataframe, with the values taking on a range from lower limit to upper limit.
- All dataframes were merged into a single dataframe.
- Outliers were capped using Winsorization, bringing down extreme values to specified percentile values.
- Data types are verified, appropriate, and consistent.
- Data is filtered to dates where only data is available across all columns.
- Necessary features such as inflation are derived.

# Exploratory Data Analysis
## Features over time
![plots_over_time](https://github.com/user-attachments/assets/86450d1f-1f41-45bd-851d-97654dc98a59)
The Federal Funds Rate and Target have been steadily decreasing since the 1980s. 

Inflation, Output Gap, and Unemployment have roughly remained around a constant level over time, despite having some sharp rises or drops.

## Distribution Plots
![dist_plots](https://github.com/user-attachments/assets/7e97e3bf-ccd3-4131-87b6-31941ba3b8c5)
Some of the features are skewed. In this case, applying transformations to the data in order to deal with skew would usually be appropriate. As Karakas makes no mention of applying transformations in their paper, this project will not apply  transformations to the data.

## Correlation Heatmap
![heatmap](https://github.com/user-attachments/assets/04f78afc-f01f-4529-9511-c6eac224169c)


Inflation, Inflation Gap, and Inflation Lag are perfectly correlated to each other, while Output Gap and Output Gap Lag (%) are very strongly correlated, which is expected. There won't be any models that use more than 1 in each set as predictors at a time, so this won't be an issue.

Unemployment and Output Gap are strongly correlated, so we will need to watch out for these when checking the Variance Inflation Factors (VIFs) for multicollinearity issues.

Target is highly correlated with our dependent variable, Federal Funds Rate, and we expect it to have the biggest impact on predictive performance of the model. Other features generally have between 40% and 50% correlation with the dependent variable, which is moderate. Unemployment has almost 0 correlation to the dependent variable which could mean low impact on predictive performance or a non-linear relationship. 

# Regression Modeling
The sets of variables for each regression model are listed as follows:

Taylor's Rule Model: 

    Dependent variable: Federal Funds Rate. 
    Independent variables: Output Gap, Inflation Gap

Karakas Model:

    Dependent variable: Federal Funds Rate. 
    Independent variables: Output Gap Lag %, Inflation Lag (%)
    
Target Model:

    Dependent variable: Federal Funds Rate. 
    Independent variables: Output Gap, Inflation Gap, Target
    
Unemployment Model:

    Dependent variable: Federal Funds Rate. 
    Independent variables: Output Gap, Inflation Gap, Unemployment
    
Both Model (uses both Unemployment and Target):

    Dependent variable: Federal Funds Rate. 
    Independent variables: Output Gap, Inflation Gap, Target, Unemployment

## Preprocessing
The data is first scaled using MinMaxScaler to remove biases from different magnitudes in numerical features.

## Fitting
The models are fitted using Ordinary Least Squares regression.

## Regression Assumptions
The assumptions of regression include:

1.	Normality of residuals
2.	Constant variance of residuals (Homoscedasticity)
3.	Independence of residuals (autocorrelation)
4.	Linearity
5.	No Multicollinearity

These assumptions can be tested using the Jarque-Bera test (Normality), Breusch-Pagan test (Homoscedasticity), Durbin-Watson test (autocorrelation), Rainbow test (Linearity), and Variance Inflation Factors (multicollinearity). A simple hypothesis test can be conducted with the null and alternative as follows:

Null Hypothesis, H0: The model does not violate the regression assumption

Alternative Hypothesis, H1: The model does violate the regression assumption

Using the 95% confidence level (significance level 0.05), the Jarque-Bera, Breusch-Pagan, and Rainbow test statistics all have p-values of around 0, which is less than the significance level. Therefore, the null, H0, would be rejected in favor of the alternative; each model violates the assumptions for these tests.

For the Durbin-Watson test, any test statistics less than 1 or greater than 3 indicate strong autocorrelation and would violate the regression assumption of independence. Since the Durbin Watson test statistics for every model lie between 0 and 0.5, the null, H0, would be rejected in favor of the alternative; each model violates the assumptions of autocorrelation.

## Assumption Statistics
|              |   Jarque-Bera p-value |   Breusch-Pagan p-value |   Durbin-Watson Test Statistic |   Rainbow Test p-value |
|:-------------|----------------------:|------------------------:|-------------------------------:|-----------------------:|
| Taylor       |                     0 |                       0 |                         0.0071 |                      0 |
| Karakas      |                     0 |                       0 |                         0.0069 |                      0 |
| Target       |                     0 |                       0 |                         0.0445 |                      0 |
| Unemployment |                     0 |                       0 |                         0.0109 |                      0 |
| Both         |                     0 |                       0 |                         0.0459 |                      0 |

All models violate the first 4 regression Assumptions. When regression assumptions are violated any metrics become biased and unreliable.

## VIFs
| Feature          |   Taylor |   Karakas |   Target |   Unemployment |     Both |
|:-----------------|---------:|----------:|---------:|---------------:|---------:|
| Inflation Gap    |   1.1778 |  nan      |   1.3848 |         1.3082 |   1.4085 |
| Inflation Lag    | nan      |    1.0887 | nan      |       nan      | nan      |
| Output Gap       |   1.1778 |  nan      |   1.1964 |         4.2165 |   5.5812 |
| Output Gap Lag % | nan      |    1.0887 | nan      |       nan      | nan      |
| Target           | nan      |  nan      |   1.2739 |       nan      |   1.702  |
| Unemployment     | nan      |  nan      | nan      |         3.68   |   4.9166 |
| const            |   5.9301 |    6.4245 |   5.9413 |        34.7962 |  43.7499 |

Any null values ('nan') in the VIF table are due to the feature not being included in the model. The threshold is that VIFs below 5 have no issues. From the VIFs, there aren't serious problems with multicollinearity except for the Output Gap in the Both Model. The best practice would be to remove that from the model entirely, but since all other regression assumptions have been violated, the regressions will be kept as is. Unemployment, the other variable of interest from the earlier heatmap, has acceptable multicollinearity levels.

## OLS Model Metrics
|              |   Mean Squared Error |   Root Mean Squared Error |   Mean Absolute Error |   Mean Percentage Error |   Mean Absolute Percentage Error |   R-Squared |
|:-------------|---------------------:|--------------------------:|----------------------:|------------------------:|---------------------------------:|------------:|
| Taylor       |                6.217 |                     2.493 |                 1.913 |                -473.933 |                          504.281 |      0.3277 |
| Karakas      |                6.574 |                     2.564 |                 2.022 |                -550.486 |                          579.147 |      0.2892 |
| Target       |                0.884 |                     0.94  |                 0.605 |                 -78.489 |                          117.299 |      0.9044 |
| Unemployment |                4.573 |                     2.138 |                 1.669 |                -416.232 |                          502.97  |      0.5056 |
| Both         |                0.864 |                     0.93  |                 0.6   |                 -85.274 |                          119.014 |      0.9066 |

The error metrics show the Taylor Model having better performance over the Karakas Model. Karakas (2023) claimed that their model had more accurate predictions, but not by much.

In this project, the Taylor model has better predictions. Note that this project uses a slightly different date range than Karakas, and this could mean that Karakas' model better predicts older data but performs poorly on more recent data. Here, the Karakas model does not perform as well as the Taylor Model, having larger average errors and percentage errors. The difference, however, is not very big, which is consistent with Karakas' findings.

All models have negative Mean Percentage Errors, and thus they all underpredict the Federal Funds Rate. The percentage error metrics seem to be very high. This is likely because due to the data having small numerical values. Average error metrics would be better for analysis.

In terms of performance, the models from worst to best are:

Karakas < Taylor < Unemployment < Target < Both

Models with lower error metrics and higher R-squared are considered better performing than others.

Adding Target or Unemployment alone to the Taylor Model increases performance, but there are marginal differences in performance when adding Unemployment alongside the Target. This suggests that Unemployment contributes little to the model, consistent with the findings from the correlation heatmap.
Note that metrics are likely biased and unreliable since regression assumptions have been violated.

## Comparison to Karakas' Paper
### Taylor vs FFR
![taylor_vs_ffr](https://github.com/user-attachments/assets/1d46e186-d0d3-4828-bd8c-aef02e2505e5)

Consistent with Karakas' plots, the Taylor Model underpredicts in year 1990, briefly overpredicts between 1990 and 1995, before it underpredicts until a bit past 2000. Around the year 2002, the model begins overpredicting for the rest of the years, except between 2005 and 2010 where it underpredicts where the actual values form a peak and around 2008 or 2009. The plots may not be an exact match, but they look similar enough to the ones shared in Karakas' paper.

### Taylor vs Karakas Model
![taylor_vs_karakas](https://github.com/user-attachments/assets/0c1cc1a1-85e4-4351-9c83-9f0a6de3148d)

There are similar patterns to the visual in Karkakas' paper for this plot as well. The Karakas Model underpredicts more than Taylor Model between 1997 to 2001 and mostly between 2010 and 2015, where there are 2 cross overs as they switch between overpredicting or underpredicting each other.

## All OLS Model Prediction Plots
![all_ols_plots](https://github.com/user-attachments/assets/450d023c-a9c5-4fb2-81a6-6445fe50276d)

The plot of predictions remains consistent with earlier findings from the metrics table. Adding Target and Unemployment improves model predictions, but they tend to more closely follow earlier years of the data.

# Neural Network (NN) Modeling
Since regression assumptions were violated, regression models are not suited for the data. Simple Neural Network models (1 hidden layer) will be used to fit each set of variables.

## NN Model Metrics
|              |   Mean Squared Error |   Root Mean Squared Error |   Mean Absolute Error |   Mean Percentage Error |   Mean Absolute Percentage Error |   R-Squared |
|:-------------|---------------------:|--------------------------:|----------------------:|------------------------:|---------------------------------:|------------:|
| Taylor       |                6.978 |                     2.642 |                 2.191 |                -635.237 |                          661.376 |      0.2455 |
| Karakas      |                7.104 |                     2.665 |                 2.168 |                -693.999 |                          715.02  |      0.2319 |
| Target       |                1.011 |                     1.005 |                 0.613 |                 -72.375 |                          129.459 |      0.8907 |
| Unemployment |                6.426 |                     2.535 |                 2.166 |                -696.731 |                          720.984 |      0.3051 |
| Both         |                1.063 |                     1.031 |                 0.657 |                -118.8   |                          156.094 |      0.8851 |

The models have returned worse results for each metric compared to the OLS models. The regression assumptions being violated have likely inflated the OLS model metrics. The model performance ranking differs slightly here. From worst performing to best performing:

Karakas < Taylor < Unemployment < Both < Target.

Like with the OLS models, there is little difference in performance between the Karakas and Taylor Models. All models have negative Mean Percentage Errors, meaning that all of the models underpredict the Federal Funds Rate.

The inclusion of both the Target and Unemployment into the Taylor Model raised the R-squared from around 0.24 to around 0.89, which is very strong performance for a model using real economic data. However, there is not much of a difference between the performance of the Target Model and the Both Model in average errors or R-squared. On the other hand, the MPE and MAPE have differences of around 27% to 40%. These values are likely high due to our data having small values. Average error metrics would be better for analysis.

Adding Unemployment to the Taylor Model alone reduces errors and explains more variance but increases percentage errors. Unemployment contributes little when added with the Target.

## NN Prediction Plots
![nn_pred_plots](https://github.com/user-attachments/assets/8d991399-8eb1-4cac-b307-9e15477d294b)

Each model tends to underpredict the Federal Funds Rate in the first half of the date range (around 1980 to 2000). Afterwards, the models tend to overpredict.

The Taylor, Karakas, and Unemployment Models tend to predict Federal Funds Rates of around 4. The Target and Both Models follow the true Federal Funds Rates more closely, having smaller gaps between them and their predictions.

# Conclusion
Although the results may differ due to different date ranges in the data, the models in this project captured similar predictive patterns to those presented in Karakas' paper from examination of the plots.

Karakas noted that the Taylor Model (and their transformation of it) did not predict the Federal Funds Rate well and mentioned their model having predictions closer to the actual Federal Funds Rate than the Taylor Model, but not by much. However, the models' metrics in this project show that it is the Taylor Model, rather than Karakas' Model that has smaller average errors and percentage errors. This is suspected to be due to how the data uses in this project contains more recent data. Also, the plot of the Taylor Model predictions does not have any FFR predictions below 0 unlike the plot in Karakas' paper. This may be because Karakas did not process outliers well, if at all.

Karakas did not mention regression assumptions or transformations in their paper when discussing their OLS models. They were checked in this project as part of the validation process. Almost all classical regression assumptions were violated, with the exception of multicollinearity. 

Karakas later created neural network models for using Taylor's Rule, getting better predictions than their OLS models. In this project, the OLS models had inflated metrics due to regression assumption violations. The neural network models, on the other hand, had lower performance metrics than our OLS models. Since Karakas got better results from their neural network model than their OLS model which should also have inflated metrics, their neural network is likely to have issues with overfitting, especially since their paper makes no mention of any techniques to counter overfitting (such as regularization, dropout, or early stopping).

This casts serious doubts about the credibility, rigor, and professionalism of Karakas' work.

For predicting the Federal Funds Rate, non-linear models should be better suited, given that no feature we used had a linear relationship with the Federal Funds Rate.

It is worth noting that the inclusion of Target and Unemployment to the Taylor Model, individually, did improve performance metrics in both the OLS and Neural Network Models. However, the Target has a much larger effect on model performance than Unemployment and Unemployment has little effect on performance when it is alongside the Target.

While performance metrics look strong for the Target Model, there is room for improvement. Some next steps would be to explore interaction terms with Unemployment or taking lags of Unemployment to avoid multicollinearity issues with Output Gap. The addition of other features such as Global Economic Growth, Prices, Interest rates, Foreign Currency Exchange Rates, Consumer Confidence, and Business Confidence are also worth exploring.

# References

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
