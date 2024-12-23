# Airbnb NYC Dataset Analysis and Predictions

## Overview
The Airbnb NYC dataset offers detailed insights into Airbnb listings across New York City in 2019. It includes information on neighborhoods, property types, pricing, availability, host details, and customer reviews. This dataset is instrumental for analyzing market trends, customer preferences, and short-term rental dynamics in one of the world's most competitive markets. It supports data-driven decision-making for optimizing pricing, identifying high-performing neighborhoods, and improving resource allocation.

By studying customer demand trends and host behavior, the analysis enhances customer experiences, refines marketing strategies, and identifies growth opportunities in underutilized areas.

## Objective
The primary goal is to utilize the Airbnb NYC dataset and machine learning models to predict growth trends in Airbnb listings and average pricing for 2019.

---

## Methodology

### 1. Methodology Overview
A systematic approach was employed, combining data cleaning, exploratory data analysis (EDA), and predictive modeling. Python served as the primary tool, leveraging libraries for data manipulation, visualization, and machine learning.

### 2. Data Collection and Preprocessing
- **Source**: Publicly available Airbnb repository.
- **Key Variables**: Room type, price, latitude, longitude, and number of reviews.
- **Preprocessing Steps**:
  - Addressed inconsistencies and missing entries.
  - Imputed null values and flagged missing categorical data.
  - Removed outliers and faulty data in numerical columns.

### 3. Exploratory Data Analysis (EDA)
- **Statistical Summaries**: Provided an overview of key metrics.
- **Visualizations**:
  - Histograms, scatterplots, and box plots to highlight trends and anomalies.
  - Scatterplots of latitude and longitude to reveal spatial trends.
  - Categorical analysis to explore room type distributions and host behavior.

### 4. Analytical and Modeling Techniques
- **Key Methods**:
  - Outlier detection using box plots.
  - Trend analysis via correlation coefficients.
  - Neighborhood grouping for average prices.
- **Predictive Modeling**:
  - Historical data (2011–2018) on inactive properties informed a polynomial model optimized with RMSE to predict inactive properties in 2019.
  - Subtracted predictions from total listings to refine the dataset.
- **Feature Engineering**: Utilized Scikit-learn’s PolynomialFeatures.
- **Evaluation Metrics**: MAE, MSE, and R².

### 5. Tools and Technologies
- **Python Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn.
- **Environment**: Jupyter Notebook for an interactive and iterative workflow.

---

## Findings

### 1. Property Types Reflect Neighborhood Trends
- **Key Insights**:
  - Northern Brooklyn and Manhattan are key activity hubs.
  - Entire homes dominate family-oriented and tourist-friendly areas.
  - Private rooms are prevalent in culturally vibrant and budget-conscious neighborhoods.
  - Shared rooms cater to affordability-driven guests.

### 2. Price Trends by Neighborhood
<p align="center">
<img src="images\airbnbtable.png"  alt="Centered Image" title="Entity-Relationship Diagram" width="500">
</p>

- **Observations**:
  - Staten Island shows unexpectedly high prices, possibly due to sampling bias or luxury listings.
  - Manhattan remains the most expensive borough overall, with an average nightly rate of $180.

### 3. Predicted Price Distribution
- Entire homes are the most expensive, reflecting demand for privacy and amenities.
- Private rooms attract solo travelers and couples, while shared rooms are budget-friendly.

---

## Visualizations
- **Figure 1**: Total number of Airbnb reviews by location.
- **Figure 2**: Distribution of Airbnb property types by location.
- **Figure 3**: Predicted average prices across boroughs.
- **Figure 4**: Spatial scatterplot of predicted Airbnb prices in NYC.

---

## Conclusion and Recommendations

### Conclusion
- Entire homes dominate tourist-heavy areas like Manhattan and Brooklyn due to demand for privacy and amenities.
- Private rooms are prevalent in budget-friendly, culturally rich neighborhoods.
- Shared rooms cater to cost-conscious travelers.
- Growth in northern Brooklyn and Queens highlights shifting market dynamics.

### Recommendations
1. **Dynamic Pricing**: Adjust pricing models based on borough-specific trends and property types.
2. **Targeted Marketing**: Focus promotional efforts in growth areas like northern Brooklyn and Queens.
3. **Data Management**: Continuously update datasets to reflect seasonal changes, outliers, and inactive listings.
4. **Customer Experience**: Enhance features in private rooms to attract solo travelers and couples.
