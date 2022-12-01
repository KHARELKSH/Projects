# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # ONLINE RETAIL BIG DATA PROJECT USING PYSPARK 

# COMMAND ----------

# MAGIC %md
# MAGIC Data description found in [this link](https://archive.ics.uci.edu/ml/datasets/Online%20Retail).

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Processing

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Data

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/OnlineRetail.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing values

# COMMAND ----------

# find missing values in all columns of the dataframe

from pyspark.sql.functions import col,isnan,when,count

df_missing = df.select([count(when(col(c).contains('None') | col(c).contains('NULL') | (col(c) == '' ) | col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns])
df_missing.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The column **CustomerID** contains **135,080** missing values. We shall drop all rows with missing values since we cannot attribute these transactions to customers - our focus in this project.

# COMMAND ----------

# drop rows with missing values in any column

df = df.na.drop()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data type conversions

# COMMAND ----------

# convert InvoiceDate to a date format

from pyspark.sql.functions import to_date

df1 = df.select('InvoiceNo','StockCode','Description','Quantity','UnitPrice','CustomerID','Country', to_date(df.InvoiceDate, 'dd/MM/yyyy HH:mm').alias('dateFormatted'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Duplicates

# COMMAND ----------

# let us keep only the distinct rows and drop any duplicates

df1 = df1.distinct()

# COMMAND ----------

# MAGIC %md
# MAGIC # Customer Segmentation

# COMMAND ----------

# MAGIC %md
# MAGIC #### Number of customers/products/countries

# COMMAND ----------

n_customers = df1.select("CustomerID").distinct().count()
n_products = df1.select("StockCode").distinct().count()
n_countries = df1.select("Country").distinct().count()

print("Number of customers: ", n_customers)
print("Number of products: ", n_products)
print("Number of countries: ", n_countries)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Products per transaction

# COMMAND ----------

from pyspark.sql.functions import count, col, desc

df1.groupBy('InvoiceNo').agg(count('StockCode').alias('no_products')).sort(desc('no_products')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transactions per country

# COMMAND ----------

from pyspark.sql.functions import count, col, desc

transactions = df1.groupBy('Country').agg(count('InvoiceNo').alias('no_transactions')).sort(desc('no_transactions'))
display(transactions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Time series of transactions per month

# COMMAND ----------

# transactions per month
from pyspark.sql.functions import month, mean, sum, asc, countDistinct

transactionsTS = df1.groupBy('dateFormatted').agg(countDistinct('InvoiceNo').alias('no_transactions')).sort(asc('dateFormatted'))
transactionsByMonth = transactionsTS.groupBy(month("dateFormatted").alias("month")).agg(sum("no_transactions").alias("transactions")).sort(asc('month'))
display(transactionsByMonth)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Canceled orders
# MAGIC - The invoice numbers of transactions that were cancelled beging with 'C' as indicated in the description of the dataset

# COMMAND ----------

cancelled = df1.filter(col("InvoiceNo").contains("C"))
display(cancelled)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that the quantities of product purchased are negative values

# COMMAND ----------

# number of orders that were cancelled

n_cancelled = cancelled.select('InvoiceNo').distinct().count()

print("Number of cancelled orders: ", n_cancelled)

# COMMAND ----------

# MAGIC %md
# MAGIC #### StockCodes with alphabets

# COMMAND ----------

# MAGIC %md
# MAGIC A look at the data shows there are stock codes that contain alphabets in them while some do not. We narrow down to this to understand more about this phenomenon.

# COMMAND ----------

# MAGIC %md
# MAGIC From the dataset we note the meaning of these stock codes as below:
# MAGIC 
# MAGIC - POST -> Postage
# MAGIC - D -> Discount
# MAGIC - C2 -> Carriage
# MAGIC - M -> Manual
# MAGIC - BANK CHARGES -> Bank Charges
# MAGIC - PADS -> Pads to match all the cushions sold
# MAGIC - DOT -> Dotcom postage

# COMMAND ----------

# get stock codes with alphabets
alphaStockCodes = df1.filter(df1.StockCode.rlike('^[a-zA-Z]'))
display(alphaStockCodes)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Basket prices

# COMMAND ----------

# we create a new df with a new column TotalPrice derived from UnitPrice and Quantity
df1 = df1.withColumn("TotalPrice", df1.UnitPrice*df1.Quantity)

mvBasketDF = df1.groupBy('CustomerID','InvoiceNo').agg(sum('TotalPrice').alias('BasketPrice'))
mvBasketDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Product Categories

# COMMAND ----------

# MAGIC %md
# MAGIC #### TF-IDF
# MAGIC - [Term frequency-inverse document frequency (TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a feature vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus. You can find more information about its implementation on PySpark [here](https://spark.apache.org/docs/latest/ml-features.html#tf-idf).

# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

tokenizer = Tokenizer(inputCol="Description", outputCol="descriptionTokens")
wordsData = tokenizer.transform(df1)

# tf transformation
hashingTF = HashingTF(inputCol="descriptionTokens", outputCol="rawFeatures", numFeatures=20)

featurizedData = hashingTF.transform(wordsData)

# alternatively, CountVectorizer can also be used to get term frequency vectors

# idf transformation
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating clusters of products

# COMMAND ----------

# Import the required libraries
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# Create an object for the Logistic Regression model
kmeans_model = KMeans(k=4)

# fit
fit_model = kmeans_model.fit(rescaledData.select('features'))

# wsse = fit_model.computeCost(final_data) for spark 2.7
wssse = fit_model.summary.trainingCost # for spark 3.0
print("The within set sum of squared error of the mode is {}".format(wssse))

# Store the results in a dataframe

results = fit_model.transform(rescaledData.select('features'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Finding optimal number of clusters
# MAGIC 
# MAGIC - **NOTE:** 
# MAGIC The part below DOES NOT scale. We could not find a suitable way of going about this step. 

# COMMAND ----------

#silhoutte analysis to find the optimal number of clusters. 
silhouette_score=[]

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features',metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2,10):  
    KMeans_model = KMeans(k=i)
    KMeans_fit=KMeans_model.fit(rescaledData)
    output=KMeans_fit.transform(rescaledData)
    score=evaluator.evaluate(output) 
    silhouette_score.append(score)    
    print("Silhouette Score:", score)

# COMMAND ----------

# MAGIC %md
# MAGIC - The silhoutte score is maximized when k = 8 but the difference between average silhoutte when k = 8 and k = 4 is very small. And, since the difference between the silhouette score is very small, Clustering with k = 8 may not group the points together that has similar characterstics than does k = 4. So, we are using clusters with 4 k's. 

# COMMAND ----------

display(results.groupby('prediction').count().sort('prediction'))

# COMMAND ----------

# MAGIC %md
# MAGIC # Customer Categories

# COMMAND ----------

# MAGIC %md
# MAGIC #### New feature: Product Category

# COMMAND ----------

# we add the product category allocations to the dataframe
from pyspark.sql.types import StructType, StructField, LongType

def with_column_index(df): 
    new_schema = StructType(df.schema.fields + [StructField("ColumnIndex", LongType(), False),])
    return df.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(schema=new_schema)

df1_ci = with_column_index(df1)
df2_ci = with_column_index(results)

join_on_index = df1_ci.join(df2_ci, df1_ci.ColumnIndex == df2_ci.ColumnIndex, 'inner').drop("ColumnIndex","features","StockCode","Quantity","UnitPrice","Country","dateFormatted")

join_on_index.sort('InvoiceNo').show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Monetary value per category

# COMMAND ----------

# create columns from product category and fill null values with 0
pivotDF = join_on_index.groupBy("CustomerID","InvoiceNo","Description").pivot("prediction").sum("TotalPrice").na.fill(value=0)
display(pivotDF)

# COMMAND ----------

mvBasketDF.sort('CustomerID').show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Value of each category per basket

# COMMAND ----------

# value distribution by product category
basketValPerCat = pivotDF.groupBy('CustomerID','InvoiceNo').agg(sum('0').alias('cat0'),sum('1').alias('cat1'),sum('2').alias('cat2'),sum('3').alias('cat3'))

# total basket value as a new column
basketValPerCat = basketValPerCat.withColumn("BasketValue", basketValPerCat.cat0 + basketValPerCat.cat1 + basketValPerCat.cat2 + basketValPerCat.cat3)

# drop rows whose total basket value is negative (ie cancellations)
basketValPerCat = basketValPerCat.where(basketValPerCat.BasketValue > 0)
basketValPerCat.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Customer data

# COMMAND ----------

from pyspark.sql.functions import count, min, sum, max, avg

customersDF = basketValPerCat.groupBy('CustomerID').agg(count('InvoiceNo').alias('no_purchases'),min('BasketValue').alias('minVal'),max('BasketValue').alias('maxVal'),\
                                                        avg('BasketValue').alias('meanVal'),sum('BasketValue').alias('totalVal'),sum('cat0').alias('cat0'),sum('cat1').alias('cat1'),sum('cat2').alias('cat2'),sum('cat3').alias('cat3'))

# COMMAND ----------

# distribution of total value of purchases per customer

display(customersDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The distribution of each individual numerical column is highly skewed probably because there exists absurdly high values related to **cancelled/erronous orders** which were not removed from the dataset. This **will** significantly affect the generizability of the model to be trained on the data.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating customer categories
# MAGIC 
# MAGIC - In this step we use unsupervised learning to create customer categories.
# MAGIC - It is the basis of the next step - classification

# COMMAND ----------

# Vector assembler is used to create a vector of input features
assembler2 = VectorAssembler(inputCols=['no_purchases','minVal','maxVal','meanVal','totalVal','cat0','cat1','cat2','cat3'],\
                            outputCol="features")

# Pipeline is used to pass the data through indexer and assembler simultaneously. Also, it helps to pre-rocess the test data
# in the same way as that of the train data.
pipe2 = Pipeline(stages=[assembler2])

final_data2=pipe2.fit(customersDF).transform(customersDF)

# Create an object for the Logistic Regression model

kmeans_model2 = KMeans(k=4)

# fit
fit_model2 = kmeans_model2.fit(final_data2.select('features'))

# wsse = fit_model.computeCost(final_data) for spark 2.7
wssse2 = fit_model2.summary.trainingCost # for spark 3.0
print("The within set sum of squared error of the mode is {}".format(wssse2))

# Store the results in a dataframe

results2 = fit_model2.transform(final_data2.select('features'))

# COMMAND ----------

display(results2.groupby('prediction').count().sort('prediction'))

# COMMAND ----------

# we add the customer category allocations to the customers dataframe
from pyspark.sql.types import StructType, StructField, LongType

def with_column_index(df): 
    new_schema = StructType(df.schema.fields + [StructField("ColumnIndex", LongType(), False),])
    return df.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(schema=new_schema)

# using the function defined earlier
df11_ci = with_column_index(customersDF)
df21_ci = with_column_index(results2)

join_on_index1 = df11_ci.join(df21_ci, df11_ci.ColumnIndex == df21_ci.ColumnIndex, 'inner').drop("ColumnIndex","features","no_purchases","minVal","maxVal","totalVal")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data visualization

# COMMAND ----------

display(join_on_index1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Classification of Customers
# MAGIC - We use only a few of the features from earlier steps in this next step to classify customers.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train-test split

# COMMAND ----------

data = join_on_index1.withColumnRenamed("prediction", "customerGroup")

train_data,test_data=data.randomSplit([0.7,0.3])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Classifier: Decision Tree

# COMMAND ----------

# Import the required libraries
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# COMMAND ----------

# Vector assembler is used to create a vector of input features
assembler = VectorAssembler(inputCols=['CustomerID', 'meanVal', 'cat0', 'cat1', 'cat2', 'cat3'],
                            outputCol="features")

# Create an object for the decision tree classifier
dt_model = DecisionTreeClassifier(labelCol='customerGroup',maxBins=5000)

pipe = Pipeline(stages=[assembler,dt_model])

# fit model
fit_model=pipe.fit(train_data)

# Store the results in a dataframe
results = fit_model.transform(test_data)

# evaluate model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

ACC_evaluator = MulticlassClassificationEvaluator(labelCol="customerGroup", predictionCol="prediction", metricName="accuracy")
accuracy = ACC_evaluator.evaluate(results)

print("The accuracy of the decision tree classifier is {}".format(accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Classifier: Random Forest

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# RandomForest model.
rf = RandomForestClassifier(labelCol="customerGroup", numTrees=10)

# Chain indexers and forest in a Pipeline
pipeRF = Pipeline(stages=[assembler, rf])

# fit model
fit_modelRF = pipeRF.fit(train_data)

# Store the results in a dataframe
resultsRF = fit_modelRF.transform(test_data)

# evaluate model
accuracyRF = ACC_evaluator.evaluate(resultsRF)

print("The accuracy of the random forest classifier is {}".format(accuracyRF))

# COMMAND ----------


