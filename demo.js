var classifier = require('./');

var bayes = new classifier.Bayesian();

bayes.train("cheap replica watches", true);
bayes.train("I don't know if this works on windows", false);

var category = bayes.classify("free watches"); 
console.log(category);  // true
