"""
@author: User_200206552
"""
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder \
        .master("local[5]") \
        .appName("Assignment_2 Question 2") \
        .config("spark.local.dir","/fastdata/acp20cvs") \
        .getOrCreate()
sc = spark.sparkContext
sc.setCheckpointDir("/fastdata/acp20cvs")  # for handling stackoverflow exception while running CV tuning
sc.setLogLevel("WARN")

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.sql import Window
import json
import time

myseed = 200206552

print("Running the code for training times using 5 CORES and the data from parquests")

# load in claim data
claims = spark.read.csv('../Data/ClaimPredictionChallenge/train_set/train_set.csv', inferSchema = "true", header = "true").cache()
claims.show(5,False)

claims.printSchema()
#claims.describe().show()

#shape of the data
print((claims.count(), len(claims.columns)))

# 1:
# Preprocessing
print("\n==========================PP================================")
claims.select(*(F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in claims.columns)).show()
#only Cat12 has na values

#replacing '?' with nan values (as we see few colums has '?')
claims.replace('?', None).select(*(F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in claims.columns)).show()
#After replacing the "?" with nan it shows Cat2, Cat4-5 and Cat7 features has more than 35% missing values so we will remove these features.

#Blind_Make, Blind_Model and Blind_Submodel have more levels so we will drop them. Row_ID, Household_ID and Vehicle are also redundant.
claims = claims.drop("Blind_Make","Blind_Model","Blind_Submodel","Row_ID","Household_ID","Vehicle","Cat2","Cat4","Cat5","Cat7")

#Removing categorical features with more classes and missing values
claims = claims.drop("Cat1","Cat6","Cat11","Cat12","OrdCat","NVCat")

#All the remaining missing data values are for categorical variables so we will remove the corresponding rows as they account for less than 0.001% of the data
claims = claims.replace("?", None)
claims = claims.dropna()

#casting the integer colums to string
for c in ['Model_Year', 'Calendar_Year']:
    claims = claims.withColumn(c, claims[c].cast(StringType()))        

#checking correlation for the numeric features
NumericColumns = [x.name for x in claims.schema.fields if x.dataType != StringType()]

corr_dict = {NumericColumns[i]:claims.stat.corr('Claim_Amount',NumericColumns[i]) for i in range(len(NumericColumns))}

sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)

#vector_col = "cor_features"
#assembler = VectorAssembler(inputCols=NumericColumns, outputCol=vector_col)
#claim_vector = assembler.transform(claims).select(vector_col)

# get correlation matrix
#claim_corr_matrix = Correlation.corr(claim_vector, vector_col)
#claim_corr_arr = claim_corr_matrix.collect()[0]["pearson({})".format(vector_col)].values
#corr_li = claim_corr_arr.reshape((17,17))[-1]

#Removing Var8 feature
claims = claims.drop("Var8")

claims.printSchema()
print((claims.count(), len(claims.columns)))


# Data Preparation

#number of rows with non-zero claim amount
claims.filter(claims.Claim_Amount !=0).count()

claims = claims.withColumn("claim",F.when(col("Claim_Amount") == 0.0, 0).otherwise(1))

#claims.show(5, False)

#Stratified Random under sampling to balance out the dataset
#based on the calculations only 0.00725 % are non-zero claims, after subtracting non-zero claims from dataset 95531 non-zero rows correspond to 0.00730 %. So, to build the model for generalized scenario instead of resampling the data to pure balanced set, sampling with slightly high proportion of zero claims than the non zero values.
 
claims = claims.sampleBy("claim", fractions={0: 0.011, 1: 1}, seed=myseed)

claims = claims.withColumn("claim", claims["claim"].cast(DoubleType()))
print("============================================================")

# 2:
# Linear Regression Model
print("\n=============================LR=============================")
# splitting the data
##(trainingDatag, testDatag) = claims.randomSplit([0.7, 0.3], myseed)

##trainingDatag.write.mode("overwrite").parquet('../Data/ClaimPredictionChallenge_training.parquet')
##testDatag.write.mode("overwrite").parquet('../Data/ClaimPredictionChallenge_test.parquet')

#train and test data from the stored parquet
trainingData = spark.read.parquet('../Data/ClaimPredictionChallenge_training.parquet').cache()
testData = spark.read.parquet('../Data/ClaimPredictionChallenge_test.parquet').cache()

trainingData.show(5, False)
testData.show(5, False)

train_count = trainingData.count()
test_count = testData.count()

#data distribution of zero and non-zero claims in train and test set
print("Train Data non-zero claims percentage: %g" % (trainingData.filter(trainingData.claim == 1.0).count() / train_count))
print("Train Data zero claims percentage: %g" % (trainingData.filter(trainingData.claim == 0.0).count() / train_count))
print("Test Data non-zero claims percentage: %g" % (testData.filter(testData.claim == 1.0).count() / test_count))
print("Test Data zero claims percentage: %g" % (testData.filter(testData.claim == 0.0).count() / test_count))

#onehotencoding for categorical features
si = StringIndexer(inputCols=['Model_Year','Calendar_Year','Cat3','Cat8','Cat9','Cat10'],
                    outputCols=['Model_Year_si', 'Calendar_Year_si', 'Cat3_si', 'Cat8_si', 'Cat9_si', 'Cat10_si'])

ohe = OneHotEncoder(inputCols=['Model_Year_si', 'Calendar_Year_si', 'Cat3_si', 'Cat8_si', 'Cat9_si', 'Cat10_si'],
                    outputCols=['Model_Year_ohe', 'Calendar_Year_ohe', 'Cat3_ohe', 'Cat8_ohe', 'Cat9_ohe', 'Cat10_ohe'])

feature_cols = ['Model_Year_ohe', 'Calendar_Year_ohe', 'Cat3_ohe', 'Cat8_ohe', 'Cat9_ohe', 'Cat10_ohe', 'Var1', 'Var2', 'Var3', 'Var4', 
                        'Var5', 'Var6', 'Var7', 'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4']

assembler_lr = VectorAssembler(inputCols = feature_cols, outputCol = 'features')

lr = LinearRegression(featuresCol='features', labelCol='Claim_Amount', regParam=0.01)

# Construct and fit pipeline to data
stages = [si, ohe, assembler_lr, lr]
pipeline = Pipeline(stages=stages)

t0 = time.time()
lr_reg = pipeline.fit(trainingData)
t1 = time.time()

#Training Time in secs
tt_lr = round(t1-t0, 3)
print("Training time for Linear Regression: %g" % tt_lr)

predictions = lr_reg.transform(testData)

#evaluator for mse
mse_evaluator = RegressionEvaluator(metricName="mse", labelCol="Claim_Amount",predictionCol="prediction")
mse = mse_evaluator.evaluate(predictions) 
print("Mean square error = " + str(mse))

#evaluator for mae
mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="Claim_Amount",predictionCol="prediction")
mae = mae_evaluator.evaluate(predictions)  
print("Mean absolute error = " + str(mae))

print("============================================================")

# 3:
# Binary Classifier + GLM
print("=============================TDM=============================")

feature_cols = ['Model_Year_si', 'Calendar_Year_si', 'Cat3_si', 'Cat8_si', 'Cat9_si', 'Cat10_si', 'Var1', 'Var2', 'Var3', 'Var4', 
                        'Var5', 'Var6', 'Var7', 'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4']

assembler_cls = VectorAssembler(inputCols = feature_cols, outputCol = 'features')

gb = GBTClassifier(labelCol="claim", featuresCol="features", seed=myseed)

stages = [si, assembler_cls, gb]
pipeline = Pipeline(stages=stages)

evaluator = MulticlassClassificationEvaluator(labelCol="claim", predictionCol="prediction", metricName="accuracy")

# Paramater grid for crossvalidation
paramGrid = ParamGridBuilder() \
    .addGrid(gb.maxDepth, [1, 5, 10]) \
    .addGrid(gb.maxIter, [10, 20, 30]) \
    .addGrid(gb.featureSubsetStrategy, ['all','sqrt', 'log2']) \
    .addGrid(gb.subsamplingRate, [0.4, 0.7, 1.0]) \
    .build()

    
#crossvalidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel_gb = crossval.fit(trainingData)
prediction = cvModel_gb.transform(testData)

def model_param(model):
  paramDict = {param[0].name: param[1] for param in model.bestModel.stages[-1].extractParamMap().items()}
  print("\nCV best model parameters:")
  print(json.dumps(paramDict, indent = 4))

#get best model parameters
model_param(cvModel_gb)

accuracy = evaluator.evaluate(prediction)
print("Accuracy for GB classifier model = %g " % accuracy)

#GBT classifier with hypertuned parameters and the default values for other parameters
gb = GBTClassifier(labelCol="claim", featuresCol="features", maxDepth= 5, maxIter=20, seed=myseed)

stages = [si, assembler_cls, gb]
pipeline = Pipeline(stages=stages)
         
t0 = time.time()
gb_cls = pipeline.fit(trainingData)
t1 = time.time()

#Training Time in secs
tt_gb = round(t1-t0, 3)
print("Training time for Gradient Boost: %g" % tt_gb)

predictions = gb_cls.transform(testData)

accuracy = evaluator.evaluate(predictions)

print("Accuracy for GB classifier model = %g " % accuracy)


#get data from predictions

#3.b

trainingData_nz = trainingData.filter(trainingData.Claim_Amount != 0.0)
trainingData_nz.cache()

glm_gamma = GeneralizedLinearRegression(featuresCol='features', labelCol='Claim_Amount', maxIter=100, regParam=0.01, tol=0.1, family='gamma', link='log')
										  
# Construct and fit pipeline to data
stages = [si, ohe, assembler_lr, glm_gamma]
pipeline = Pipeline(stages=stages)

t0 = time.time()
glm_gamma_nz = pipeline.fit(trainingData_nz)
t1 = time.time()

#Training Time in secs
tt_ga = round(t1-t0, 3)
print("\nTraining time for GLM - Gamma: %g" % tt_ga)


# tandem model

prediction_cls = gb_cls.transform(testData)

#removing standard indexer columns as the si stage is present in glm as well
prediction_cls = prediction_cls.drop('Model_Year_si', 'Calendar_Year_si', 'Cat3_si', 'Cat8_si', 'Cat9_si', 'Cat10_si', 'features')

predictions = glm_gamma_nz.transform(prediction_cls.withColumnRenamed("prediction","claim_pred"))										  
										  
predictions = predictions.withColumn("Claim_Amount_pred", F.when(predictions["claim_pred"] == 0.0, 0.0).otherwise(predictions["prediction"]))

#evaluator for mse
mse = mse_evaluator.evaluate(predictions) 
print("Mean square error = " + str(mse))

#evaluator for mae
mae = mae_evaluator.evaluate(predictions)  
print("Mean absolute error = " + str(mae))

print("============================================================")
              
              