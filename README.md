# pyspark-template

This pyspark Term project for MET CS 777 by Sushant Mohan Khot to analyze and segment Customers based on available Credit Card data.


## Describe here your project

The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. 

The file consists of customer credit card data with 18 behavioral variables of the customer.

The description of features / columns of Credit Card dataset :-

**CUST_ID** : Identification of Credit Card holder (Categorical)

**BALANCE** : Balance amount left in their account to make purchases

**BALANCE_FREQUENCY** : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)

**PURCHASES** : Amount of purchases made from account

**ONEOFF_PURCHASES** : Maximum purchase amount done in one-go

**INSTALLMENTS_PURCHASES** : Amount of purchase done in installment

**CASH_ADVANCE** : Cash in advance given by the user

**PURCHASES_FREQUENCY** : How frequently the Purchases are being made, score between 0 and 1 
(1 = frequently purchased, 0 = not frequently purchased)
**ONEOFF_PURCHASES_FREQUENCY** : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)

**PURCHASES_INSTALLMENTS_FREQUENCY** : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)

**CASH_ADVANCE_FREQUENCY** : How frequently the cash in advance being paid

**CASH_ADVANCE_TRX** : Number of Transactions made with "Cash in Advanced"

**PURCHASES_TRX** : Number of purchase transactions made

**CREDIT_LIMIT** : Limit of Credit Card for user

**PAYMENTS** : Amount of Payment done by user

**MINIMUM_PAYMENTS** : Minimum amount of payments made by user

**PRC_FULL_PAYMENT** : Percent of full payment paid by user

**TENURE** : Tenure of credit card service for user


We will answer the below Reasearch Questions:

**Research Questions:**

1. Can we try to segment Customers into Clusters to identify which group is spending high Amount on Purchases using their Credit Cards?

2. Are there any Customers that have High Credit Limit but are not spending high on Purchases using their credit Cards?

3. Are their any Customers that have High Balance available and are NOT spending much on Purchases using their Credit Cards?

4. How Accurately are our Clustering Models identifying these Customer groups?

We ran KMeans and GMM clustering Models on our dataset.

We also did split our dataset into Training (70%) and Testing (30%). We have also combined the cluster labels to 2 major classes:
0 = Customers with Low Credit / Balance and Low Purchases. (Negative Event)
1 = Customers with Med-High Credit / Balance and Med-High Purchases. (Positive Event)

After Training the “Train” dataset, we use the model on our “Test” dataset and find out the Class / Labels and Probabilities for the KMeans and GMM Models.

We Calculated the Confusion Matrix and the different Scores for both the Models based on out Test dataset outcomes:

**KMeans Model:**

*Accuracy: 0.971

*Precision: 0.998

*Recall: 0.965

*F1_Score: 0.981


**GMM Model:**

*Accuracy: 0.274

*Precision: 0.746

*Recall: 0.302

*F1_Score: 0.43


# Submit your python scripts .py 

My term project consists of a single .py file. Hence I have added the project in main_task1.py

# Other Documents. 

I attached my presentation file in pdf format which explains in detail about the project, processes followed and outputs.

I have also added the output Plots as .png files. 

Finally, I have added the dataset used in .csv format.


# How to run  

Run the task 1 by submitting the task to spark-submit. 


```python

spark-submit main_task1.py docs\CC_data.csv

```




