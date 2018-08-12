# Project_Mcnulty - Instacard Market Basket Analysis
This is the [Instacart Market Basket Analysis project from Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis) with some modification. Instead of predicting what the customer will buy in the next order, I am trying to predict whether they will reorder the products they've ordered in the past. 

### Feature List
1. User Features  
    - Total order number
    - Total product number
    - Average cart size
    - Percentage of orders on weekend vs weekday
    - Average days between orders
    - Number of different product the user ever purchased
    - Percentage of products that the user purchased only once among all the products
2. Product Features
    - Total product reorder count
    - Total counts of order on this product
    - Averge product add to cart order 
    - Percentage of reorder among total orders
    - Department
    - Aisle
3. User-Product Features
    - How many times the user reorder the product in the past
    - How many orders ago is the last time the user ordered that product
    - Days since the user last time ordered the product
    - Average days since the prior order each time the user orders the product
    - Average add to cart order over scaled within each user
    - Average add to cart order
    - Percentage of the user's orders containing the product
    - Average days betweeen the customer order the product.
    - Ratio between average days the user purchase a product vs days since last order
    - Has the user order the same product that day
4. Order Specific
    - Order day of the week
    - Order hour of the day
    - Days since last order

### Model and Hyperparameters
Gradient Boosting (XGB)
- n_estimators = 339
- max_depth = 4
- learning_rate = 0.025
- subsample = 0.7
- colsample = 0.6
- Probability threshold = 0.2138
- Metrics - f1 score 

### Results 
Average f1 on cross validation of subset (2.5%, 0.2M records): 0.429 (std. 0.014)

F1 on holdout set: 0.401


