# Digit Recognizers

## Colin's stuff

### get_data.py

This file pulls down the data set for you.  I made a Kaggle user called AustinData.  Email me for the password.  Data is downloaded to a local folder called `data`.

### analyzer.py
This contains the main machine learning that is going on.  I use two classes so far -- the `FileHandler` will grab the data (there is an optional argument for how many lines of data you grab), and the `Pixel` class is meant to do processing on individual images.  For example, the `Pixel` class has a method `self.features()` which will return an array of all the features extracted.  This is then aggregated in the `FileHandler` as `self.features()`, which returns a matrix of the features from each `Pixel`.

There is also a function `normalize` which will normalize the features.  A good example use of this class is

```
#!python

data = get_data.get_data_files()
files = analyzer.FileHandler(data['train'])
files.show_bad_classifiers()
```

`get_data_files()` will pull down a dictionary with train and test data sets.  The `FileHandler` will train itself, report on how it did, and print a 4 x 4 grid of examples it got wrong.  The oob (out of bag) score is a way to not need a cross validation or test set when using a random forest -- it keeps track of which data points went into which decision trees, and excludes those decision trees when predicting for a data point.  

The current highest score I've seen is 96.22% correct, which uses each pixel as a feature, as well as the mean of the pixels (`Pixel.ink()`), which I think should help discriminate between a 1 and an 8, for example.
