from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors

sc = SparkContext()

#load data file
ts = MLUtils.loadLibSVMFile(sc, "train_scaled.txt")

lrw = LinearRegressionWithSGD.train(ts, iterations=100, intercept=True)
print("weights: " ,lrw.weights)
print("intercept: " ,lrw.intercept)

#Initialize vector
prediction = Vectors.dense(0.343158,0.762712,-0.27243,0.678756,-0.0348259,0.163445,-0.00635324,-0.295056,-0.396509,0.142857,-0.256757,-0.118812,0.294964,0.821429)
ts_predict = lrw.predict(prediction)
print(ts_predict)
