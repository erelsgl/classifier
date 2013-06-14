var _ = require("underscore")._;

/**
 * Test the given classifier on the given train-set and test-set.
 * @param classifierConst identifies the type of classifier to test.
 * @param options set of options for initializing that classifier.
 * @param trainSet, testSet arrays with objects of the format: {input: "sample1", output: "class1"}
 * @return an object with the results of the test (- number of classification errors, and elapsed time).
 */
function testPartition(classifierConst, options, trainSet, testSet) {
  var classifier = new classifierConst(options);

  var beginTrain = Date.now();

  classifier.trainAll(trainSet);

  var beginTest = Date.now();

  var error = classifier.test(testSet);

  var endTest = Date.now();

  return {
    error : error,
    trainTime : beginTest - beginTrain,
    testTime : endTest - beginTest,
  };
}

/**
 * Test the given classifier on the given dataset.
 * @param classifierConst identifies the type of classifier to test.
 * @param options set of options for initializing that classifier.
 * @param data the gold-standard: an array with objects of the format: {input: "sample1", output: "class1"}
 * @param k number of folds for cross-validation.
 * 
 * @return an object with the results of the test (- number of classification errors, and elapsed time).
 */
module.exports = function crossValidate(classifierConst, options, data, k) {
  k = k || 3;
  var size = data.length / k;

  // Shuffle the gold-standard dataset:
  data = _(data).sortBy(function(num){
    return Math.random();
  });

  var avgs = {
    error : 0,
    trainTime : 0,
    testTime : 0,
  };

  var results = _.range(k).map(function(i) {
    var dclone = _(data).clone();
    var testSet = dclone.splice(i * size, size);
    var trainSet = dclone;

    var result = testPartition(classifierConst, options, trainSet, testSet);

    _(avgs).each(function(sum, i) {
      avgs[i] = sum + result[i];
    });
  });

  _(avgs).each(function(sum, i) {
    avgs[i] = sum / k;
  });

  avgs.testSize = size;
  avgs.trainSize = data.length - size;

  return avgs;
}