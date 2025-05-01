# Introduction
In this project we will construct models for the Federal Funds Effective Rate. This is a rate determined by the market, similar to stock prices in the stock market. The Federal Funds Rate is a key tool used in monetary policy that influences economic activity. The Federal Reserve sets a Target for the Federal Funds Rate, then performs operations such as trading bonds to adjust the Federal Funds Rate, bringing it closer to the Target. Predicting the Federal Funds Rate benefits educators, economists, investors, financial institutions, and policy planners.

This project aims to first reproduce regression models predicting the Federal Funds Rate using Taylor’s Rule, a policy guideline by John Taylor from Stanford in 1993 and a modification of this used by researcher Alper D. Karakas, the equations of which are derived in Karkas’ (2023) paper, “Reevaluating the Taylor Rule with Machine Learning.”

We will then attempt to construct other models by adding the Target and Unemployment Rate to the Taylor Model to see if the addition of new features can improve the performance of regression models in predicting the Federal Funds Effective Rate. We choose to build off the Taylor Model as this is the foundational model and Karakas (2023) found little difference in performance between this model and their model.

We will also check the assumptions of regression for each model and construct simple Neural Network Models for the regression equations.

# Jupyter Notebook
1.	Download Python, Jupyter Notebook, and necessary packages (Matplotlib, Numpy, Pandas, Plotly, Scikit-Learn, Seaborn, Statsmodels, and Tensorflow)
2.	Download the FFR_modeling.ipynb file and save it on to your machine
3.	Open Jupyter Notebook, move and open the file FFR_regression_modeling.ipynb
4.	From the toolbar at the top, click on “Run,” then click on “Run all cells.”

# Streamlit Application Link
Access the Streamlit Web Application with the following URL:
https://ffr-modeling-5n85wvsg3gdnvd4z4yz3kh.streamlit.app/
