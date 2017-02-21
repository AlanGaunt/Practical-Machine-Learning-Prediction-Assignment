
**Practical Machine Learning Prediction Assignment**
====================================================

### The goal of this project is to predict the manner in which 6 participants perform specific exercises (AKA the "classe" variable in the training set). To complete the predicitons, we will test both a *Classification Tree* and *Random Forest* model to determine the best fit.

**Prediction preperation**
--------------------------

``` r
## Checking to ensure required packages are installed

require(caret)
```

    ## Loading required package: caret

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
require(rattle)
```

    ## Loading required package: rattle

    ## Rattle: A free graphical interface for data mining with R.
    ## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
require(rpart)
```

    ## Loading required package: rpart

``` r
require(rpart.plot)
```

    ## Loading required package: rpart.plot

``` r
require(randomForest)
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
## Downloading data file(s)

trainURL = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

## Loading data into memory and identifying NAs

trainRaw = read.csv(url(trainURL), na.strings = c("NA","#DIV/0!",""))

testRaw = read.csv(url(testURL), na.strings = c("NA","#DIV/0!",""))

## Partioning the Training data to enable cross validation. 60% of the data will be use for training, 40% for testing. 

set.seed(1)
trainPart = createDataPartition(trainRaw$classe, p = .60, list = FALSE)
inTrain = trainRaw[trainPart, ]
inTest = trainRaw[-trainPart, ]

## Subsetting partioned data for more accurate fit

NZV = nearZeroVar(trainRaw)
inTrain = inTrain[, -NZV]
inTest = inTest[, -NZV]

NAs = sapply(inTrain, function(x) mean(is.na(x))) > 0.95
inTrain = inTrain[, NAs == FALSE]
inTest = inTest[, NAs == FALSE]

inTrain  = inTrain[,-(1:5)]
inTest = inTest[,-(1:5)]

## Subsetting full training set for final fit

trainSub = trainRaw[, -NZV]
testSub = testRaw[, -NZV]

zNAs = sapply(trainSub, function(x) mean(is.na(x))) > 0.95
trainSub = trainSub[, zNAs == FALSE]
testSub = testSub[, zNAs == FALSE]

trainSub  = trainSub[,-(1:5)]
testSub = testSub[,-(1:5)]
```

**Classification Tree**
-----------------------

### The first model tested is a Classification Tree given the outcomes are categorical and processing time is fairly quick.

``` r
## Building the Classification Tree model

set.seed(1)

modFit = train(classe ~ ., data = inTrain, method = "rpart")
fancyRpartPlot(modFit$finalModel)
```

![](PML_Course_Project_vGitHub_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r
Predictions = predict(modFit, inTest)
confusionMatrix(inTest$classe, Predictions)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2014   36  177    0    5
    ##          B  611  517  390    0    0
    ##          C  625   41  702    0    0
    ##          D  534  234  457    0   61
    ##          E  162  112  306    0  862
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.5219         
    ##                  95% CI : (0.5108, 0.533)
    ##     No Information Rate : 0.5029         
    ##     P-Value [Acc > NIR] : 0.0003988      
    ##                                          
    ##                   Kappa : 0.3766         
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.5104  0.55000  0.34547       NA   0.9289
    ## Specificity            0.9441  0.85505  0.88545   0.8361   0.9162
    ## Pos Pred Value         0.9023  0.34058  0.51316       NA   0.5978
    ## Neg Pred Value         0.6559  0.93315  0.79469       NA   0.9897
    ## Prevalence             0.5029  0.11981  0.25899   0.0000   0.1183
    ## Detection Rate         0.2567  0.06589  0.08947   0.0000   0.1099
    ## Detection Prevalence   0.2845  0.19347  0.17436   0.1639   0.1838
    ## Balanced Accuracy      0.7272  0.70253  0.61546       NA   0.9225

### The accuracy of the Classification Tree is unsastifactory and could need be used to confidently complete our predictions.

**Random Forest**
-----------------

### The second model tested is a Random Forest given the Classification Tree was largely inaccurate.

``` r
## Building the Random Forest model

set.seed(1)

fitControl = trainControl(method = "cv", number = 3)

fit = train(classe ~ ., data = inTrain, method = "rf", trControl = fitControl)

pred = predict(fit, inTest)

confusionMatrix(inTest$classe, pred)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2232    0    0    0    0
    ##          B    1 1514    2    1    0
    ##          C    0    2 1363    3    0
    ##          D    0    0    6 1280    0
    ##          E    0    0    0    4 1438
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9976          
    ##                  95% CI : (0.9962, 0.9985)
    ##     No Information Rate : 0.2846          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9969          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9996   0.9987   0.9942   0.9938   1.0000
    ## Specificity            1.0000   0.9994   0.9992   0.9991   0.9994
    ## Pos Pred Value         1.0000   0.9974   0.9963   0.9953   0.9972
    ## Neg Pred Value         0.9998   0.9997   0.9988   0.9988   1.0000
    ## Prevalence             0.2846   0.1932   0.1747   0.1642   0.1833
    ## Detection Rate         0.2845   0.1930   0.1737   0.1631   0.1833
    ## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      0.9998   0.9990   0.9967   0.9964   0.9997

### The accuracy of the Random Forest is signifcantly better than the Classifcation Tree and can be confidently used to complete our predicitions. It's important to note that the out-of-sample error will most likely be higher when applied to the Test set. To reduce the significance, we'll train the model using the full training set and therefore increase our prediction accuracy.

``` r
finalFit = train(classe ~ ., data = trainSub, method = "rf", trControl = fitControl)

finalPred = predict(finalFit, testSub)

print(finalPred)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

**Results**
-----------

### **The final predicitions of this model were 100% accurate when applied to the Project Quiz.**
