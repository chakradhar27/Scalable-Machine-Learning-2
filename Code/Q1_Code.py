"""
@author: User_200206552
"""
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np
import pandas as pd
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import json
import time

spark = SparkSession.builder \
        .master("local[10]") \
        .appName("Assignment_2 Question 1") \
        .config("spark.local.dir","/fastdata/acp20cvs") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

higgsdata = spark.read.csv('../Data/HIGGS.csv.gz', sep=',', inferSchema='true').cache()

higgsdata.first()
higgsdata.printSchema()    #all the columns are infered to be double type so no need to convert the column type

#adding the column names manually as the header is missing (replacing spaces with "_")
higgsdata_columns = ['label', 'lepton pT', 'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

new_column_list= list(map(lambda x: x.replace(" ", "_"), higgsdata_columns))

higgsdata = higgsdata.toDF(*new_column_list)

#higgsdata.printSchema()

print("Running the code for training times using 10 CORES and the data from parquests (sanity check) by commenting out the crossvalidation part")

# 1:
# Model building and cross-validation on the subset of the dataset

#subset of the large dataset - commented the code inorder to fetch the data from parquet
##higgsdata_sample=higgsdata.sample(False,0.01, 200206552).cache()

##(trainingDatag, testDatag) = higgsdata_sample.randomSplit([0.7, 0.3], 200206552)

#trainingDatag.write.mode("overwrite").parquet('../Data/higgsdata_subset_training.parquet')
#testDatag.write.mode("overwrite").parquet('../Data/higgsdata_subset_test.parquet')

#subset train and test data from the stored parquet
trainingData_sub = spark.read.parquet('../Data/higgsdata_subset_training.parquet').cache()
testData_sub = spark.read.parquet('../Data/higgsdata_subset_test.parquet').cache()

print(f"There are {trainingData_sub.count()} rows in the training set, and {testData_sub.count()} in the test set")

vecAssembler = VectorAssembler(inputCols=new_column_list[1:], outputCol="features")

# Evaluator for both the crossvalidation, and later for checking accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

#Model Parameters
def model_param(model):
  paramDict = {param[0].name: param[1] for param in model.bestModel.stages[-1].extractParamMap().items()}
  print("\nCV best model parameters:")
  print(json.dumps(paramDict, indent = 4))
   
## Randon forest
print("=======================RF_Sample===========================")
rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=200206552)

stages = [vecAssembler, rf]
pipeline = Pipeline(stages=stages)

# Paramater grid for crossvalidation
paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [1, 5, 10]) \
    .addGrid(rf.maxBins, [10, 20, 32]) \
    .addGrid(rf.numTrees, [20, 50, 100]) \
    .addGrid(rf.featureSubsetStrategy, ['all','sqrt','log2']) \
    .addGrid(rf.subsamplingRate, [0.1, 0.5, 0.9]) \
    .build()

    
#crossvalidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel_rf = crossval.fit(trainingData_sub)
prediction = cvModel_rf.transform(testData_sub)

#get best model parameters
model_param(cvModel_rf)

# Accuracy
accuracy = evaluator.evaluate(prediction)
print("Accuracy for best rf model = %g " % accuracy)

# Compute raw scores on the test set
predictionAndLabels = prediction.rdd.map(lambda x: (float(x.prediction), x.label))

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under ROC curve
print("Area under ROC for best rf model = %g" % metrics.areaUnderROC)
print("===========================================================")


## Gradient Boost
print("=======================GB_Sample===========================")
gb = GBTClassifier(labelCol="label", featuresCol="features", seed=200206552)

stages = [vecAssembler, gb]
pipeline = Pipeline(stages=stages)

# Paramater grid for crossvalidation
paramGrid = ParamGridBuilder() \
    .addGrid(gb.maxDepth, [1, 5, 10]) \
    .addGrid(gb.maxBins, [10, 20, 32]) \
    .addGrid(gb.maxIter, [10, 20, 30]) \
    .addGrid(rf.featureSubsetStrategy, ['all','sqrt','log2']) \
    .addGrid(gb.subsamplingRate, [0.1, 0.5, 0.9]) \
    .build()

#crossvalidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel_gb = crossval.fit(trainingData_sub)
prediction = cvModel_gb.transform(testData_sub)

#get best model parameters
model_param(cvModel_gb)

# Accuracy
accuracy = evaluator.evaluate(prediction)
print("Accuracy for best gb model = %g " % accuracy)

# Compute raw scores on the test set
predictionAndLabels = prediction.rdd.map(lambda x: (float(x.prediction), x.label))

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under ROC curve
print("Area under ROC for best gb model = %g" % metrics.areaUnderROC)
print("===========================================================")

## Neural network
print("=======================NN_Sample===========================")
mpc = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", maxIter=100, seed=200206552)

stages = [vecAssembler, mpc]
pipeline = Pipeline(stages=stages)

#Number of input features
input_features = len(trainingData_sub.columns)-1


# Paramater grid with the different number of layers and nodes in each layer for crossvalidation
#choosing the nodes for hidden layers as per the general principle that the size can be between the input layer and output layer
paramGrid = ParamGridBuilder() \
            .addGrid(mpc.layers, [[input_features,20,10,2],
                                  [input_features,50,20,10,2],
                                  [input_features,50,20,2],
                                  [input_features,80,40,20,2]]) \
            .addGrid(mpc.solver, ['l-bfgs', 'gd']) \
            .addGrid(mpc.stepSize, [0.03, 0.05, 0.1]) \
            .build()

    
#crossvalidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel_nn = crossval.fit(trainingData_sub)
prediction = cvModel_nn.transform(testData_sub)

#get best model parameters
model_param(cvModel_nn)

# Accuracy
accuracy = evaluator.evaluate(prediction)
print("Accuracy for best gb model = %g " % accuracy)

# Compute raw scores on the test set
predictionAndLabels = prediction.rdd.map(lambda x: (float(x.prediction), x.label))

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under ROC curve
print("Area under ROC for best gb model = %g" % metrics.areaUnderROC)
print("===========================================================")

# 2:
# Model building on the whole dataset

#commented the code inorder to fetch the data from parquet
(trainingDatag, testDatag) = higgsdata.randomSplit([0.7, 0.3], 200206552)

##trainingDatag.write.mode("overwrite").parquet('../Data/higgsdata_training.parquet')
##testDatag.write.mode("overwrite").parquet('../Data/higgsdata_test.parquet')

#train and test data from the stored parquet
trainingData = spark.read.parquet('../Data/higgsdata_training.parquet').cache()
testData = spark.read.parquet('../Data/higgsdata_test.parquet').cache()

print(f"There are {trainingData.count()} rows in the training set, and {testData.count()} in the test set")

## Random Forest
print("=======================RF===========================")
rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=10, maxBins=20, numTrees=100, \
          featureSubsetStrategy="all", subsamplingRate=0.9, seed=200206552)

stages = [vecAssembler, rf]
pipeline = Pipeline(stages=stages)

t0 = time.time()
pipelineModel = pipeline.fit(trainingData)
t1 = time.time()

#Training Time in secs
tt_rf = round(t1-t0, 3)
print("Training time for Random Forest: %g" % tt_rf)

predictions = pipelineModel.transform(testData)

featureImp_rf = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), pipelineModel.stages[-1].featureImportances)),
  columns=["feature", "importance"])
featureImp_rf.sort_values(by="importance", ascending=False)

accuracy = evaluator.evaluate(predictions)

print("Accuracy for RF model = %g " % accuracy)

# Compute raw scores on the test set
predictionAndLabels = predictions.rdd.map(lambda x: (float(x.prediction), x.label))

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under ROC curve
print("Area under ROC for RF model = %g" % metrics.areaUnderROC)
print("===========================================================")

## Gradient Boost
print("=======================GB===========================")
gb = GBTClassifier(labelCol="label", featuresCol="features", seed=200206552, \
              maxDepth=5, maxBins=32, maxIter=30, stepSize=0.1, subsamplingRate=0.9 )

stages = [vecAssembler, gb]
pipeline = Pipeline(stages=stages)

t0 = time.time()
pipelineModel = pipeline.fit(trainingData)
t1 = time.time()

#Training Time in secs
tt_gb = round(t1-t0, 3)
print("Training time for Gradient Boost: %g" % tt_gb)

predictions = pipelineModel.transform(testData)

featureImp_gb = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), pipelineModel.stages[-1].featureImportances)),
  columns=["feature", "importance"])
featureImp_gb.sort_values(by="importance", ascending=False)

accuracy = evaluator.evaluate(predictions)

print("Accuracy for GB model = %g " % accuracy)

# Compute raw scores on the test set
predictionAndLabels = predictions.rdd.map(lambda x: (float(x.prediction), x.label))

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under ROC curve
print("Area under ROC for GB model = %g" % metrics.areaUnderROC)
print("===========================================================")

## Neural network
print("=======================NN===========================")
layers = [len(trainingData.columns)-1, 20, 10, 2] 
mpc = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", maxIter=100, layers=layers, seed=200206552)

stages = [vecAssembler, mpc]
pipeline = Pipeline(stages=stages)

t0 = time.time()
pipelineModel = pipeline.fit(trainingData)
t1 = time.time()

#Training Time in secs
tt_nn = round(t1-t0, 3)
print("Training time for Neural Network: %g" % tt_nn)

predictions = pipelineModel.transform(testData)

accuracy = evaluator.evaluate(predictions)
print("Accuracy for NN model= %g " % accuracy)

# Compute raw scores on the test set
predictionAndLabels = predictions.rdd.map(lambda x: (float(x.prediction), x.label))

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under ROC curve
print("Area under ROC for NN model = %g" % metrics.areaUnderROC)
print("===========================================================")

#Training Time
tt_m = tt_rf + tt_gb + tt_nn 
print("\nTraining time of all three models using 10 CORES: %g" % tt_m)

# 3:
# Most Relevant Features
#Random Forest
print("===========================================================")
print("Three most relevant features from RF model:")
print(featureImp_rf.iloc[:3])
print("===========================================================")

#Gradient Boost
print("===========================================================")
print("Three most relevant features from GB model:")
print(featureImp_gb.iloc[:3])
print("===========================================================")

 