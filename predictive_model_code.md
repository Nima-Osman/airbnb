# Explanation

In our analysis, we initially aimed to determine how many properties were added over the years. However, we were unable to achieve this because the `last_review` variable only indicated how many properties became "inactive" each year, essentially showing how many properties were no longer listed on Airbnb annually. Consequently, we have learnt that each year, some properties stop being active on the platform.

The year 2019 contains the highest number of properties, as it includes both inactive and active listings. However, we do not yet know how many of those properties will eventually become inactive. We are, however, certain that this will happen, as it has in previous years. Therefore, our new objective is to estimate how many properties will become inactive in 2019. By doing so, we can remove these listings from our dataset and focus on identifying the active ones.

This approach will allow us to reduce our overall sample size with a reliable, transparent, and trustworthy filter. We will concentrate solely on our active data. To achieve this, we grouped all properties according to the years of their last review up to 2018, creating a smaller dataset that illustrates how many properties became inactive each year from 2011 to 2018. We did not include the last year, 2019, in this calculation because we want to predict how many properties will become inactive that year based on historical data.

To find this answer, we will apply machine learning techniques. After estimating the number of inactive properties for 2019, we will subtract this predicted number from the total number of properties in 2019. By doing this, we can approximate the true number of active properties. A smaller sample size will result in more reliable answers, as it allows us to focus more easily on the most significant data.

For our machine learning approach, we utilise a polynomial model, as it is the most appropriate for our data. However, before applying the polynomial model, we first need to determine the optimal degree that will allow the model to fit our data effectively. We employ a reliable assessment method to identify the best degree for our model, ensuring that it is neither overfitted nor underfitted. To achieve this, we use the RMSE (Root Mean Square Error) as our evaluation metric.


#Our target is to find the ACTIVE and most Reliable properties
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Reading our dataset file
df = pd.read_csv("AB_NYC_2019.csv")

#Converting the last_review column to a Datetime format
df['last_review'] = pd.to_datetime(df['last_review'])
df['last_review'] = df['last_review'].dt.year

#Replacing a value in a column
df['room_type'] = df['room_type'].replace('Entire home/apt', 'Entire home')

#Renaming columns
df = df.rename(columns={"neighbourhood_group": "Place", "neighbourhood": "Hood", "latitude": "Lat", "longitude": "Lot", "room_type": "Type", "price": "Price",'last_review':'Year'})

#Dropping rows with N/A values
df = df.dropna()


#Resetting our index number series everytime after changes 
df = df.reset_index(drop=True)

#Finding how many rows and columns our dataset has
df.shape

(38821, 16)

#How to filter out a dataset
df_2018=df[df['Year'] < 2019]
df_2018.shape

(13620, 16)

#Depicting the expansion of inactive properties over the years
sns.scatterplot(df_2018, x = 'Lot', y = 'Lat', hue = 'Year')
 


#Counting every Place for each Year
df_ml = df_2018.groupby('Year').agg(
    Count_Places=('Place', 'count')
).reset_index()


df_ml = df_ml[['Year', 'Count_Places']]
df_ml

        Year      Count_Places
0  2011.0             7
1  2012.0            25
2  2013.0            48
3  2014.0           199
4  2015.0          1388
5  2016.0          2703
6  2017.0          3203
7  2018.0          6047





#Applying Polynomial Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

#Preparing our data
X = df_ml[['Year']]
y = df_ml[['Count_Places']]

#Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Storing our results
degrees = np.arange(1, 10)
test_scores = []
train_scores = []

for degree in degrees:
    
    #Transforming the data to include polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    #Fitting the polynomial regression model
    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    #Evaluating the model
    predictions_train = model.predict(X_poly_train)
    predictions_test = model.predict(X_poly_test)

    #Calculating RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, predictions_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, predictions_test))

    train_scores.append(train_rmse)
    test_scores.append(test_rmse)

#Plotting RMSE scores for training and testing
plt.plot(degrees, train_scores, label='Training RMSE', marker='o')
plt.plot(degrees, test_scores, label='Testing RMSE', marker='o')
plt.title('Polynomial Degree vs. RMSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.legend()
plt.show()






#Choosing the best degree based on RMSE scores
best_degree = degrees[np.argmin(test_scores)]
print(f'The best polynomial degree is: {best_degree}')


 


The best polynomial degree is: 2


Explanation:
A degree of 2 indicates a quadratic model, which fits the data well without overcomplicating it.
When the degree is larger than 2, the training RMSE becomes lower than the testing RMSE, indicating overfitting. This means that the model has learned the noise in the training data rather than the underlying pattern.



#Fitting the final model with the best degree
poly = PolynomialFeatures(degree=best_degree)
X_poly_full = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly_full, y)


#Predictions and error metrics
predictions_train = model.predict(X_poly_full)
metrics = {
    'Mean Absolute Error': mean_absolute_error(y, predictions_train),
    'Mean Squared Error': mean_squared_error(y, predictions_train),
    'Root Mean Squared Error': mean_squared_error(y, predictions_train, squared=False),
    'R Square Score': r2_score(y, predictions_train)
}

for metric, value in metrics.items():
    print(f'{metric}: {value:.8f}')



#Predicting for the year 2019
year_2019 = pd.DataFrame({'Year': [2019]})
year_2019_poly = poly.transform(year_2019)
predictions_2019 = model.predict(year_2019_poly)

predicted_df = pd.DataFrame(predictions_2019, columns=['Count_Places'])
predicted_df['Year'] = 2019

#Adding the prediction to the original DataFrame
df_ml = pd.concat([df_ml, predicted_df], ignore_index=True)

#Plotting original and predicted data
plt.scatter(X, y, color='blue', label='Original data')
plt.scatter(predicted_df['Year'], predicted_df['Count_Places'], color='red', label='Predicted data (2019)')
plt.title('Polynomial Regression Prediction')
plt.xlabel('Year')
plt.ylabel('Count of Places')
plt.legend()
plt.show()

print(df_ml)


Mean Absolute Error: 255.40178567
Mean Squared Error: 97763.31844181
Root Mean Squared Error: 312.67126258
R Square Score: 0.97623694

 

     Year  Count_Places
0  2011.0      7.000000
1  2012.0     25.000000
2  2013.0     48.000000
3  2014.0    199.000000
4  2015.0   1388.000000
5  2016.0   2703.000000
6  2017.0   3203.000000
7  2018.0   6047.000000
8  2019.0   7935.749999



#Machine Learning predicted that we will have 7936 inactive properties in the year of 2019.
#So, we have to remove the exact number of rows in the year of 2019 RANDOMLY,
#in order to keep only the active ones.

#Removing the Predicted number of inactive properties Randomly
df_2019 = df[df['Year'] == 2019] 

if len(df_2019) >= 7936:
    predicted_values_remove = df_2019.sample(n=7936, random_state=1)
    df_2019 = df_2019.drop(predicted_values_remove.index)
else:
    print("Not enough rows to remove 7936 entries.")

df_2019 = df_2019.dropna()
df_2019 = df_2019.sort_values(by=['Year'], ascending=False)
df_2019 = df_2019.reset_index(drop=True)
df_2019.shape

(17265, 16)


#Applying all the appropriate filters (We review all the histograms and the distribution of those numeric variables)

#We want properties that are available at least one day of the year 
df_2019 = df_2019[df_2019['minimum_nights'] < 365]

#We do not need properties with lists that do not progress
df_2019 = df_2019[df_2019['calculated_host_listings_count'] < 50]

#We want properties that works and have at least 1 review
df_2019 = df_2019[df_2019['number_of_reviews'] > 0]

#The same as above. We want active properties
df_2019 = df_2019[df_2019['reviews_per_month'] > 0]

#We want available and active properties. We do not want properties at pause
df_2019 = df_2019[df_2019['availability_365'] > 0]

#Properties that are available more than 365 days are faulty data
df_2019 = df_2019[df_2019['availability_365'] < 365]

#Properties with zero value price do not exist. Those are also faulty data
df_2019 = df_2019[df_2019['Price'] > 0]

#We do not want rows with N/A numbers
df_2019 = df_2019.dropna()

#That is how we reset our indexing numbers after every change in our data
df_2019 = df_2019.reset_index(drop=True)

#The shape gives as the number of rows and columns in a table
df_2019.shape

(14492, 16)
#Removing all the outliers from numerical column variables

Q1 = df_2019['minimum_nights'].quantile(0.25)
Q3 = df_2019['minimum_nights'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_2019 = df_2019[(df_2019['minimum_nights'] >= lower_bound) & (df_2019['minimum_nights'] <= upper_bound)]

Q1 = df_2019['reviews_per_month'].quantile(0.25)
Q3 = df_2019['reviews_per_month'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_2019 = df_2019[(df_2019['reviews_per_month'] >= lower_bound) & (df_2019['reviews_per_month'] <= upper_bound)]

Q1 = df_2019['calculated_host_listings_count'].quantile(0.25)
Q3 = df_2019['calculated_host_listings_count'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_2019 = df_2019[(df_2019['calculated_host_listings_count'] >= lower_bound) & (df_2019['calculated_host_listings_count'] <= upper_bound)]

Q1 = df_2019['number_of_reviews'].quantile(0.25)
Q3 = df_2019['number_of_reviews'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_2019 = df_2019[(df_2019['number_of_reviews'] >= lower_bound) & (df_2019['number_of_reviews'] <= upper_bound)]

Q1 = df_2019['Price'].quantile(0.25)
Q3 = df_2019['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_2019 = df_2019[(df_2019['Price'] >= lower_bound) & (df_2019['Price'] <= upper_bound)]

df_2019 = df_2019.reset_index(drop=True)

df_2019.shape

(9735, 16)











#How to count a categorical variable
df_2019['Place'].value_counts()

Place
Brooklyn         4205
Manhattan        3710
Queens           1391
Bronx             306
Staten Island     123

#How to check out for outliers in a numerical variable
sns.boxplot(x=df_2019['Price'])
plt.title('Price Boxplot')
plt.xlabel('Price')
plt.grid()
plt.show()

 


#How to view a numerical variable distribution with Histogram
sns.histplot(df_2019['Price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid()
plt.show()

 





#How to find out the average Prices by Place
avg_prices_places = df_2019.groupby('Place')['Price'].mean().reset_index()
avg_prices_places = avg_prices_places.sort_values(by='Price', ascending=False)

ax = sns.barplot(x='Price', y='Place', data=avg_prices_places, palette='viridis')
ax.set_title('Average Prices by Place')
ax.set_xlabel('Average Prices')
ax.set_ylabel('Places')
plt.show()

 




#How to find out the average Prices by Type of property
avg_prices_types = df_2019.groupby('Type')['Price'].mean().reset_index()
avg_prices_types = avg_prices_types.sort_values(by='Price', ascending=False)

ax_T = sns.barplot(x='Price', y='Type', data=avg_prices_types, palette='viridis')
ax_T.set_title('Average Prices by Type')
ax_T.set_xlabel('Average Prices')
ax_T.set_ylabel('Types')
plt.show()

 










#How to find out the Total Reviews per Place
total_reviews = df_2019.groupby('Place')['number_of_reviews'].sum().reset_index()

sns.barplot(x='Place', y='number_of_reviews', data=total_reviews, palette='viridis')
plt.title('Total Reviews Per Place')
plt.xlabel('Place')
plt.ylabel('Total Reviews')
plt.show()

 



#How to find out the Total number of properties by Place
type_counts_place = df_2019.groupby(['Place', 'Type']).size().reset_index(name='Count')

sns.barplot(data=type_counts_place, x='Place', y='Count', hue='Type')
plt.title('Count of Property Types by Place')
plt.ylabel('Count')
plt.xlabel('Place')
plt.legend(title='Type')
plt.show()

 

df_2019 = df_2019.reset_index(drop=True)

df_2019.shape

(9735, 16)




#Top 10 hoods with higher reviews
top_review_hoods = df_2019.groupby(['Hood','Place'])['number_of_reviews'].sum().reset_index()
top_review_hoods = top_review_hoods.sort_values(by='number_of_reviews', ascending=False)
print(top_review_hoods.head(10))

                  Hood      Place  number_of_reviews
12   Bedford-Stuyvesant   Brooklyn              33582
90               Harlem  Manhattan              25825
201        Williamsburg   Brooklyn              24913
27             Bushwick   Brooklyn              15299
91       Hell's Kitchen  Manhattan              13265
59          East Harlem  Manhattan              11627
49        Crown Heights   Brooklyn              11181
62         East Village  Manhattan              10814
189     Upper East Side  Manhattan               9265
190     Upper West Side  Manhattan               8246


#Top 10 hoods with the most properties with their average prices
top_property_hoods = df_2019.groupby(['Hood', 'Place']).agg(count=('Hood', 'size'), average_price=('Price', 'mean')).reset_index()
top_property_hoods = top_property_hoods.sort_values(by='count', ascending=False)
print(top_property_hoods.head(10))

                 Hood      Place  count  average_price
12   Bedford-Stuyvesant   Brooklyn    864     108.643519
201        Williamsburg   Brooklyn    772     135.865285
90               Harlem  Manhattan    639     113.215962
27             Bushwick   Brooklyn    453      92.450331
91       Hell's Kitchen  Manhattan    367     174.651226
62         East Village  Manhattan    324     163.194444
49        Crown Heights   Brooklyn    313     117.578275
59          East Harlem  Manhattan    281     129.387900
189     Upper East Side  Manhattan    278     151.377698
190     Upper West Side  Manhattan    242     162.260331


#Top 10 hoods with the most properties with their Type
top_type_hoods = df_2019.groupby(['Hood', 'Place', 'Type']).size().unstack().fillna(0)
top_type_hoods['Total'] = top_type_hoods[['Entire home', 'Private room', 'Shared room']].sum(axis=1)
top_type_hoods = top_type_hoods.sort_values(by='Total', ascending=False)
print(top_type_hoods.head(10))

Type                          Entire home  Private room  Shared room  Total
Hood               Place                                                   
Bedford-Stuyvesant Brooklyn         475.0         380.0          9.0  864.0
Williamsburg       Brooklyn         412.0         353.0          7.0  772.0
Harlem             Manhattan        266.0         359.0         14.0  639.0
Bushwick           Brooklyn         160.0         291.0          2.0  453.0
Hell's Kitchen     Manhattan        234.0         128.0          5.0  367.0
East Village       Manhattan        221.0          99.0          4.0  324.0
Crown Heights      Brooklyn         188.0         120.0          5.0  313.0
East Harlem        Manhattan        138.0         136.0          7.0  281.0
Upper East Side    Manhattan        180.0          91.0          7.0  278.0
Upper West Side    Manhattan        142.0          95.0          5.0  242.0
#Top 10 most expensive hoods 
expensive_hoods = df_2019.groupby(['Hood','Place'])['Price'].mean().reset_index()
expensive_hoods = expensive_hoods.sort_values(by='Price', ascending=False)
print(expensive_hoods.head(10))

                 Hood          Place       Price
89         Grymes Hill  Staten Island  300.000000
32   Castleton Corners  Staten Island  299.000000
123         Mill Basin       Brooklyn  299.000000
133           Neponsit         Queens  237.000000
185            Tribeca      Manhattan  220.266667
137               NoHo      Manhattan  214.750000
88   Greenwich Village      Manhattan  203.838710
72   Flatiron District      Manhattan  197.777778
197       West Village      Manhattan  195.411290
20        Breezy Point         Queens  195.000000


#Depicting the expansion of property prices
sns.scatterplot(df_2019, x = 'Lot', y = 'Lat', hue = 'Price')

 

#Depicting the expansion of property types
sns.scatterplot(df_2019, x = 'Lot', y = 'Lat', hue = 'Type')

 
