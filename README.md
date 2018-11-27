# Project Mcnulty: Instacard Market Basket Analysis
This is the [Instacart Market Basket Analysis project from Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis) with some modification. Instead of predicting what the customer will buy in the next order, I am trying to predict whether they will reorder the products they've ordered in the past. 

### Feature List
1. User Features  
    - Total order count
    - Total product count
    - Average cart size
    - Average days between orders
    - Number of unique product the user ever purchased
    - Percentage of products that the user purchased only once among all the products
2. Product Features
    - Total counts of order on this product
    - Averge product add to cart order 
    - Order percentage
    - Binned department
    - Binned Aisle
3. User-Product Features
    - How many times the user reordered the product in the past
    - How many orders ago is the last time the user ordered that product
    - Average days between the user ordered the product
    - Average add to cart order over scaled within each user
    - Percentage of the user's orders containing the product
    - Days between orders last time the user purchased the product
4. Order Specific
    - Days since last product order vs average
    - Whether ordered the product the same day
    - Days since last order
    - Deviation of the days since prior order to the mean

### Model and Hyperparameters
Gradient Boosting (XGB)
- n_estimators = 843
- max_depth = 6
- learning_rate = 0.01
- subsample = 0.3
- colsample = 0.7
- Probability threshold = 0.2114
- Metrics - f1 score 

### Results 
Average f1 on cross validation of subset (6.3%, 1.9M records): 0.437 (std. 0.0078)

F1 on holdout set: 0.436

#### Note
The python scripts were used to prepare dataset, train the model and make prediction. Model selection and feature engineering were performed on a subset of data using notebook. When using the script, put raw data in the raw_data fold and create data and result fold to save results and the data sets. 


