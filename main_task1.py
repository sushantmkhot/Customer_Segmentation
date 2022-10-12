"""
@author: Sushant Khot
Class: CS 777 - Fall 1
Date: 10/12/2021
Term Project : Customer Segmentation by Credit Card data using Unsupervised Learning (KMeans and GMM)
"""

'''
Introduction:
The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. 
The file consists of customer credit card data with 18 behavioral variables of the customer.

The description of features / columns of Credit Card dataset :-

CUST_ID : Identification of Credit Card holder (Categorical)
BALANCE : Balance amount left in their account to make purchases
BALANCE_FREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
PURCHASES : Amount of purchases made from account
ONEOFF_PURCHASES : Maximum purchase amount done in one-go
INSTALLMENTS_PURCHASES : Amount of purchase done in installment
CASH_ADVANCE : Cash in advance given by the user
PURCHASES_FREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
ONEOFF_PURCHASES_FREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
PURCHASES_INSTALLMENTS_FREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
CASH_ADVANCE_FREQUENCY : How frequently the cash in advance being paid
CASH_ADVANCE_TRX : Number of Transactions made with "Cash in Advanced"
PURCHASES_TRX : Number of purchase transactions made
CREDIT_LIMIT : Limit of Credit Card for user
PAYMENTS : Amount of Payment done by user
MINIMUM_PAYMENTS : Minimum amount of payments made by user
PRC_FULL_PAYMENT : Percent of full payment paid by user
TENURE : Tenure of credit card service for user

Research Questions:
1. Can we try to segment Customers into Clusters to identify which group is spending high Amount on Purchases using their Credit Cards?
2. Are there any Customers that have High Credit Limit but are not spending high on Purchases using their credit Cards?
3. Are their any Customers that have High Balance available and are NOT spending much on Purchases using their Credit Cards?
4. How Accurately are our Clustering Models identifying these Customer groups?

'''

# ==========================================================================================
#                              Import all required Libraries
# ==========================================================================================

from __future__ import print_function
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
import matplotlib.pyplot as plt

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA
from pyspark.sql.functions import udf


# ==========================================================================================
#                        Initialize Spark Context and Spark Session
# ==========================================================================================

sc = SparkContext(appName="TermProject")
spark = SparkSession.builder.master("local[*]").appName("SparkTermProject").getOrCreate()
sqlContext = SQLContext(sc)


# ==========================================================================================
#                           Read the data into a dataframe
# ==========================================================================================

# cc_df = spark.read.csv("C://Sushant//CS777//TermProject//data//CC_data.csv", header=True, inferSchema=True)
cc_df = spark.read.csv(sys.argv[1], header=True, inferSchema=True)

# Checking the schema of the dataframe
print("Schema of the dataframe after import: \n")
print(cc_df.printSchema())
print("\n")

print("Showing some records from the dataset: \n")
print(cc_df.show(3))
print("\n")

# Total 8950 rows in the dataframe
print("Total number of rows in the dataset: ", str(cc_df.count()))
print("\n")

# We will now drop all rows with missing data
cc_df = cc_df.na.drop()

# The total number of rows are now 8636
print("Total number of rows after removing missing data: ", str(cc_df.count()))
print("\n")


# ==========================================================================================
#                           KMeans - Unsupervised learning
# ==========================================================================================

'''
K-Means is an unsupervised machine learning algorithm that groups data into k number of clusters. 
The number of clusters (k) is user-defined and the algorithm will try to group the data even if this number is not optimal for the specific case. 
'''

'''
We now combine the input columns from the dataset that will be used for analysis using VectorAssembler.
VectorAssembler is a transformer that combines a given list of columns into a single vector column. 
It is useful for combining raw features and features generated by different feature transformers into a single feature vector.
'''

vecAssembler = VectorAssembler(inputCols=[
 'BALANCE',
 'BALANCE_FREQUENCY',
 'PURCHASES',
 'ONEOFF_PURCHASES',
 'INSTALLMENTS_PURCHASES',
 'CASH_ADVANCE',
 'PURCHASES_FREQUENCY',
 'ONEOFF_PURCHASES_FREQUENCY',
 'PURCHASES_INSTALLMENTS_FREQUENCY',
 'CASH_ADVANCE_FREQUENCY',
 'CASH_ADVANCE_TRX',
 'PURCHASES_TRX',
 'CREDIT_LIMIT',
 'PAYMENTS',
 'MINIMUM_PAYMENTS',
 'PRC_FULL_PAYMENT',
 'TENURE'], outputCol="features")

features_cc_df = vecAssembler.transform(cc_df)

print("Vector Assembled 'features column' in the Spark dataframe: \n")
print(features_cc_df.show(2))

'''
We will now scale the "features" column using StandardScalar.
StandardScaler transforms a dataset of Vector rows, normalizing each feature to have unit standard deviation and/or zero mean.
'''

cc_scale = StandardScaler(inputCol='features', outputCol='standardized')
cc_data_scale = cc_scale.fit(features_cc_df)
cc_data_scale_output = cc_data_scale.transform(features_cc_df)

print("\n")
print("Scaled 'standardized column' in the Spark dataframe: \n")
print(cc_data_scale_output.show(2))
print("\n")

# ==========================================================================================
#                           Dimensionality reduction using PCA
# ==========================================================================================

'''
Dimensionality reduction is the process of reducing the number of variables under consideration. 
It can be used to extract latent features from raw and noisy features or compress data while maintaining the structure.
We will apply tis to our dataframe on the "standardized" column to reduce it to a vector of 2 elements from 17 using PCA.
Principal component analysis (PCA) is a statistical method to find a rotation such that the first coordinate has the largest variance possible, 
and each succeeding coordinate, in turn, has the largest variance possible. 
The columns of the rotation matrix are called principal components. PCA is used widely in dimensionality reduction.
'''
pca = PCA(k=2, inputCol="standardized")

pca.setOutputCol("pca_features")
pca_model = pca.fit(cc_data_scale_output)
pca_model.getK()

reduced_features_pca_df = pca_model.transform(cc_data_scale_output)

print("Reduced features 'pca_features column' in the Spark Dataframe: \n")
print(reduced_features_pca_df.show(2))
print("\n")

'''
Before we start with KMeans algorithm for clustering, we will need to analyze the data to understand the optimal K to be used.
We will approach this problem in 2 different ways:
1. Calculating Silhouette Score
2. Elbow method. 
'''

'''
Silhouette score:
Silhouette score is used to evaluate the quality of clusters created using clustering algorithms such as K-Means in terms of 
how well samples are clustered with other samples that are similar to each other. 
The Silhouette score is calculated for each sample of different clusters.
'''

# silhouette Score
silhouette_score_pca = []
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='pca_features', metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2, 10):
    KMeans_algo_pca = KMeans(featuresCol='pca_features', k=i, seed=0)

    KMeans_fit_pca = KMeans_algo_pca.fit(reduced_features_pca_df)

    output_pca = KMeans_fit_pca.transform(reduced_features_pca_df)

    score_pca = evaluator.evaluate(output_pca)

    silhouette_score_pca.append(score_pca)

    print("Silhouette Score:", score_pca)


# Visualizing the silhouette scores in a plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(2, 10), silhouette_score_pca)
plt.title('KMeans: Silhouette Score')
ax.set_xlabel('k')
ax.set_ylabel('cost')
plt.show()

'''
K = 3 is maximum Silhouette Score.
'''


'''
The Elbow Method:
The Elbow method is a very popular technique and the idea is to run k-means clustering for a range of clusters k (let’s say from 1 to 10) and for each value, 
we are calculating the sum of squared distances from each point to its assigned center(distortions).
When the distortions are plotted and the plot looks like an arm then the “elbow”(the point of inflection on the curve) is the best value of k.
'''

# ELBOW Calculate cost and plot
cost_pca = np.zeros(10)

for k in range(2, 10):
    kmeans_pca = KMeans().setK(k).setSeed(0).setFeaturesCol('pca_features')
    kmeans_model_pca = kmeans_pca.fit(reduced_features_pca_df)
    cost_pca[k] = kmeans_model_pca.summary.trainingCost


cost_pca = cost_pca[2:]
new_col = [2, 3, 4, 5, 6, 7, 8, 9]


# import pylab as pl
plt.plot(new_col, cost_pca)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('KMeans: Elbow Curve')
plt.show()

'''
We observe that K = 3 is our Optimal value.
'''

# Train the KMeans model:
K = 3


def train_kmeans(K, train_df):
    kmeans = KMeans().setK(K).setSeed(0).setFeaturesCol('pca_features')
    kmeans_model = kmeans.fit(train_df)
    return kmeans_model


kmeans_model_pca = train_kmeans(3, reduced_features_pca_df)
centers_pca = kmeans_model_pca.clusterCenters()
transformed_kmeans_df = kmeans_model_pca.transform(reduced_features_pca_df)

transformed_kmeans_df = transformed_kmeans_df.withColumn("Label", transformed_kmeans_df.prediction)
transformed_kmeans_df = transformed_kmeans_df.drop("prediction")


print("The K = 3 Centroids from the KMeans mode: ", centers_pca)
print("\n")
print("Dataframe with predicted column renamed to 'Label':")
print(transformed_kmeans_df.show(3))
print("\n")

'''
Assign clusters to events:
We will now assign the individual rows to the nearest cluster centroid. 
That can be done with the transform method, which adds 'prediction' column to the dataframe. 
The prediction value is an integer between 0 and k, but it has no correlation to the y value of the input.
'''

centers_pca_x = [centers_pca[0][0], centers_pca[1][0], centers_pca[2][0]]
centers_pca_y = [centers_pca[0][1], centers_pca[1][1], centers_pca[2][1]]

'''
Now we have the Customer IDs assigned to their individual clusters.
We can visualize this by putting this in a pandas df.
'''

kmeans_df_label_pca = transformed_kmeans_df.toPandas().set_index('CUST_ID')

plt.scatter(x=kmeans_df_label_pca.CREDIT_LIMIT, y=kmeans_df_label_pca.PURCHASES, c=kmeans_df_label_pca.Label)
plt.scatter(x=centers_pca_x, y=centers_pca_y, c="red")
plt.title("Kmeans Clustering - Customers Credit Limit vs Purchases")
plt.xlabel('Credit Limit')
plt.ylabel('Purchases')
plt.show()

'''
From the plot we see there are 3 clusters w.r.t Credit Limit vs Purchases.
cluster 0: yellow : Customers with Low to High Credit Limit and Low Amount of Purchases on their credit cards.
cluster 1: purple: Medium to High Credit Limit but Low Amount of Purchases on their credit cards.
cluster 2: green: Low to High Credit Limit and High Amount of Purchases on their credit cards.
'''


plt.scatter(x=kmeans_df_label_pca.BALANCE, y=kmeans_df_label_pca.PURCHASES, c=kmeans_df_label_pca.Label)
plt.scatter(x=centers_pca_x, y=centers_pca_y, c="red")
plt.title("Kmeans Clustering - Customers Balance Amount vs Purchases")
plt.xlabel('Balance Amount')
plt.ylabel('Purchases')
plt.show()

'''
From the plot we see there are 3 clusters w.r.t Balance vs Purchases.
cluster 0: yellow : Customers with Low Balance and Low Amount of Purchases on their credit cards
cluster 1: purple: Customers with Low to High Balance but Low Amount of Purchases on their credit cards.
cluster 2: green: Customers with Low to High Balance but High Amount of Purchases on their credit cards.
'''


# ==========================================================================================
#                    GMM (Gaussian Mixture Model)  - Unsupervised learning
# ==========================================================================================

'''
A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.
After Training the model, we can see “probability” column added to the dataframe.
'''

def train_gmm(K, train_df):
    gmm = GaussianMixture(featuresCol="pca_features").setK(K).setSeed(0)
    gmm_model = gmm.fit(train_df)
    return gmm_model

gmm_model_pca = train_gmm(3, reduced_features_pca_df)
transformed_gmm_df = gmm_model_pca.transform(reduced_features_pca_df)

transformed_gmm_df = transformed_gmm_df.withColumn("Label", transformed_gmm_df.prediction)
transformed_gmm_df = transformed_gmm_df.drop("prediction")

print("Gaussian Dataframe: \n")
print(gmm_model_pca.gaussiansDF.show(2))
print("\n")

gmm_df_label_pca = transformed_gmm_df.toPandas().set_index('CUST_ID')

plt.scatter(x=gmm_df_label_pca.CREDIT_LIMIT, y=gmm_df_label_pca.PURCHASES, c=gmm_df_label_pca.Label)
plt.title("GMM Clustering - Customers Credit Limit vs Purchases")
plt.xlabel('Credit Limit')
plt.ylabel('Purchases')
plt.show()


'''
From the plot we see there are 3 clusters w.r.t Credit Limit vs Purchases.
cluster 0: yellow: Low Amount of purchases done by Customers having low to high Credit Limit
cluster 1: green: Low to Medium amount of purchases done by Customers having low to high Credit Limit
cluster 2: purple: Medium to high Amount of Purchases done by Customers having low to high Credit Limit
'''

plt.scatter(x=gmm_df_label_pca.BALANCE, y=gmm_df_label_pca.PURCHASES, c=gmm_df_label_pca.Label)
plt.title("GMM Clustering - Customers Balance Amount vs Purchases")
plt.xlabel('Balance Amount')
plt.ylabel('Purchases')
plt.show()

'''
From the plot we see there are 3 clusters w.r.t Balance vs Purchases.
cluster 0: yellow: Low Amount of Purchases done by Customers having low to high Balance on their credit cards.
cluster 1: green: Low to Medium amount of Purchases done by Customers having low Balance on their credit cards.
cluster 2: purple: Medium to high Amount of Purchases done by Customers having low to high Balance on their credit cards.
'''


# ==========================================================================================
#                           Testing our Unsupervised Models
# ==========================================================================================

'''
The below function "col_func" takes in value existing Labels from the dataframe as input and returns the Labels to 0 or 1 based on the condition.
Condition if existing Label == 0 then return 0 else return 1.
0 = Customers with Low Credit / Balance and Low Purchases. (Negative Event)
1 = Customers with Med-High Credit / Balance and Med-High Purchases. (Positive Event).
'''

def col_func(label):
    if label == 0:
        return 0
    else:
        return 1


func_udf = udf(col_func)


# ==========================================================================================
#           Training and Testing with KMeans model (Training 70% and Testing 30%)
# ==========================================================================================

df_split_kmeans = transformed_kmeans_df.randomSplit([0.7, 0.3], seed=6118)

train_kmeans_df = df_split_kmeans[0]
test_kmeans_df = df_split_kmeans[1]

train_kmeans_df = train_kmeans_df.withColumn('New_Label', func_udf(train_kmeans_df['Label']))
test_kmeans_df = test_kmeans_df.withColumn('New_Label', func_udf(test_kmeans_df['Label']))

# Train the df
kmeans_train_model = train_kmeans(3, train_kmeans_df)

# Test using training model
test_kmeans_prediction_df = kmeans_train_model.transform(test_kmeans_df)

test_kmeans_prediction_df = test_kmeans_prediction_df.withColumn('New_Prediction', func_udf(test_kmeans_prediction_df['prediction']))

print("KMeans Test Dataframe with predicted values: \n")
print(test_kmeans_prediction_df.show(3))
print("\n")


# Confusion Matrix and Accuracy calculations:

TP_kmeans = test_kmeans_prediction_df.filter((test_kmeans_prediction_df.New_Label == test_kmeans_prediction_df.New_Prediction) & (test_kmeans_prediction_df.New_Prediction == 1)).count()
TN_kmeans = test_kmeans_prediction_df.filter((test_kmeans_prediction_df.New_Label == test_kmeans_prediction_df.New_Prediction) & (test_kmeans_prediction_df.New_Prediction == 0)).count()
FP_kmeans = test_kmeans_prediction_df.filter((test_kmeans_prediction_df.New_Label == 0) & (test_kmeans_prediction_df.New_Prediction == 1)).count()
FN_kmeans = test_kmeans_prediction_df.filter((test_kmeans_prediction_df.New_Label == 1) & (test_kmeans_prediction_df.New_Prediction == 0)).count()

accuracy_kmeans = (TP_kmeans + TN_kmeans) / (TP_kmeans + TN_kmeans + FP_kmeans + FN_kmeans)
precision_kmeans = (TP_kmeans) / (TP_kmeans + FP_kmeans)
recall_kmeans = (TP_kmeans) / (TP_kmeans + FN_kmeans)
f1_score_kmeans = (2 * (recall_kmeans * precision_kmeans)) / (recall_kmeans + precision_kmeans)

print("Accuracy of KMeans Model: ", str(round(accuracy_kmeans, 3)))
print("\n")
print("Precision Score of KMeans Model: ", str(round(precision_kmeans, 3)))
print("\n")
print("Recall Score of KMeans Model: ", str(round(recall_kmeans, 3)))
print("\n")
print("F1 Score of KMeans Model: ", str(round(f1_score_kmeans, 3)))
print("\n")


# ==========================================================================================
#           Training and Testing with GMM model (Training 70% and Testing 30%)
# ==========================================================================================

df_split_gmm = transformed_gmm_df.randomSplit([0.7, 0.3], seed=6118)

train_gmm_df = df_split_gmm[0]
test_gmm_df = df_split_gmm[1]

train_gmm_df = train_gmm_df.drop("probability")
test_gmm_df = test_gmm_df.drop("probability")

train_gmm_df = train_gmm_df.withColumn('New_Label', func_udf(train_gmm_df['Label']))

# Train the df
gmm_train_model = train_gmm(3, train_gmm_df)

# Test using training model
test_gmm_prediction_df = gmm_train_model.transform(test_gmm_df)

test_gmm_prediction_df = test_gmm_prediction_df.withColumn('New_Label', func_udf(test_gmm_prediction_df['Label']))
test_gmm_prediction_df = test_gmm_prediction_df.withColumn('New_Prediction', func_udf(test_gmm_prediction_df['prediction']))


print("GMM Test Dataframe with predicted values: \n")
print(test_gmm_prediction_df.show(3))
print("\n")


# Confusion Matrix and Accuracy calculations:

TP_gmm = test_gmm_prediction_df.filter((test_gmm_prediction_df.New_Label == test_gmm_prediction_df.New_Prediction) & (test_gmm_prediction_df.New_Prediction == 1)).count()
TN_gmm = test_gmm_prediction_df.filter((test_gmm_prediction_df.New_Label == test_gmm_prediction_df.New_Prediction) & (test_gmm_prediction_df.New_Prediction == 0)).count()
FP_gmm = test_gmm_prediction_df.filter((test_gmm_prediction_df.New_Label == 0) & (test_gmm_prediction_df.New_Prediction == 1)).count()
FN_gmm = test_gmm_prediction_df.filter((test_gmm_prediction_df.New_Label == 1) & (test_gmm_prediction_df.New_Prediction == 0)).count()

accuracy_gmm = (TP_gmm + TN_gmm) / (TP_gmm + TN_gmm + FP_gmm + FN_gmm)
precision_gmm = (TP_gmm) / (TP_gmm + FP_gmm)
recall_gmm = (TP_gmm) / (TP_gmm + FN_gmm)
f1_score_gmm = (2 * (recall_gmm * precision_gmm)) / (recall_gmm + precision_gmm)

print("Accuracy of GMM Model: ", str(round(accuracy_gmm, 3)))
print("\n")
print("Precision Score of GMM Model: ", str(round(precision_gmm, 3)))
print("\n")
print("Recall Score of GMM Model: ", str(round(recall_gmm, 3)))
print("\n")
print("F1 Score of GMM Model: ", str(round(f1_score_gmm, 3)))
print("\n")


# ==========================================================================================
#                                   Final Observations
# ==========================================================================================
'''
Accuracy: In terms of Accuracy, KMeans model is a lot Accurate in clustering our Customers based on their Credit Card data compared to GMM. 
In our case, the Accuracy of GMM is very low compared to KMeans.

Precision: It is a score that tells us: Out of all the positive predicted, what percentage is truly positive (Med-High Purchase Customers). 
In this case we see that again KMeans has performed better than compared to GMM.

Recall: It is a score that tells us: Out of the total positive, what percentage are predicted positive (Med-High Purchase Customers). 
In this case we see that KMeans has performed much better than compared to GMM.

F1 Score: It is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. 
Intuitively, it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. 
In this case we see that KMeans has performed much better than compared to GMM.
In our case, KMeans has outperformed GMM.
'''

# ==========================================================================================
#                           Research Questions – Answered below:
# ==========================================================================================
'''
1) Can we try to segment Customers into Clusters to identify which group is spending high Amount on Purchases using their Credit Cards? 
Yes, we were able to cluster the customers based on their credit card data to find who are spending High Amount on Purchases.

2) Are there any Customers that have High Credit Limit but are not spending high on Purchases using their credit Cards?
Yes, we were able to cluster the customers based on their credit card data to find Customers with high Credit Limit and not spending much on Amount on Purchases.

3) Are their any Customers that have High Balance available and are NOT spending much on Purchases using their Credit Cards?
Yes, we have found a cluster of Customers who have high Balance available on the Credit Card and not spending much on Purchases.

4) How Accurately are our Clustering Models identifying these Customer groups? 
KMeans model has more accurately segmented our customers than GMM based on their credit card data. We have calculated their Accuracies and other scores.
'''


# Free up the Spark Context
spark.stop()