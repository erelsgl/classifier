var _ = require("underscore")._;

var Bayesian = function(options) {
  options = options || {}
  this.thresholds = options.thresholds || {};
  this.default = options.default || 'unclassified';
  this.weight = options.weight || 1;
  this.assumed = options.assumed || 0.5;

  var backend = options.backend || { type: 'memory' };
  switch(backend.type.toLowerCase()) {
    case 'redis':
      this.backend = new (require("./backends/redis").RedisBackend)(backend.options);
      break;
    case 'localstorage':
      this.backend = new (require("./backends/localStorage")
                     .LocalStorageBackend)(backend.options);
      break;
    default:
      this.backend = new (require("./backends/memory").MemoryBackend)();
  }
}

Bayesian.prototype = {
  getCats : function(callback) {
    return this.backend.getCats(callback);
  },

  getWordCounts : function(words, cats, callback) {
    return this.backend.getWordCounts(words, cats, callback);
  },

  incDocCounts : function(samples, callback) {
    // accumulate all the pending increments
    var wordIncs = {};
    var catIncs = {};
    samples.forEach(function(sample) {
      var cat = sample.cat;
      catIncs[cat] = catIncs[cat] ? catIncs[cat] + 1 : 1;

      var words = this.getFeatures(sample.doc);
      words.forEach(function(word) {
        wordIncs[word] = wordIncs[word] || {};
        wordIncs[word][cat] = wordIncs[word][cat] ? wordIncs[word][cat] + 1 : 1;
      }, this);
    }, this);

    return this.backend.incCounts(catIncs, wordIncs, callback);
  },

  setThresholds : function(thresholds) {
    this.thresholds = thresholds;
  },

  getFeatures : function(doc) {
    if (_(doc).isArray()) {
      return doc;
    } else if (_(doc).isString()) {
    	var words = doc.split(/\W+/);
    	return _(words).uniq();
    } else if (_(doc).isObject()) {
    	return Object.keys(doc);
    } else {
    	console.dir(doc);
    	throw new Error("Unknown document type");
    }
  },

  /**
   * Tell the classifier that the given document belongs to the given category.
   * @param doc [string] a training sample.
   * @param cat [string] the correct class of this sample.
   */
  train : function(doc, cat, callback) {
    this.incDocCounts([{doc: doc, cat: cat}], function(err, ret) {
      if (callback) {
        callback(ret);
      }
    });
  },

  /**
   * Train the classifier with all the given documents.
   * @param data an array with objects of the format: {input: sample1, output: class1}
   */
  trainAll : function(data, callback) {
    data = data.map(function(item) {
      return {doc: item.input, cat: item.output};
    });
    this.incDocCounts(data, function(err, ret) {
      if (callback) {
        callback(ret);
      }
    });
  },

  wordProb : function(word, cat, cats, counts) {
    // times word appears in a doc in this cat / docs in this cat
    var prob = (counts[cat] || 0) / cats[cat];

    // get weighted average with assumed so prob won't be extreme on rare words
    var total = _(cats).reduce(function(sum, p, cat) {
      return sum + (counts[cat] || 0);
    }, 0, this);
    return (this.weight * this.assumed + total * prob) / (this.weight + total);
  },

  getCatProbs : function(cats, words, counts) {
    var numDocs = _(cats).reduce(function(sum, count) {
      return sum + count;
    }, 0);

    var probs = {};
    _(cats).each(function(catCount, cat) {
      var catProb = (catCount || 0) / numDocs;

      var docProb = _(words).reduce(function(prob, word) {
        var wordCounts = counts[word] || {};
        return prob * this.wordProb(word, cat, cats, wordCounts);
      }, 1, this);

      // the probability this doc is in this category
      probs[cat] = catProb * docProb;
    }, this);
    return probs;
  },

  getProbs : function(doc, callback) {
    var that = this;
    this.getCats(function(cats) {
      var words = that.getFeatures(doc);
      that.getWordCounts(words, cats, function(counts) {
        var probs = that.getCatProbs(cats, words, counts);
        callback(probs);
      });
    });
  },

  /**
   * Used for classification.
   * Get the probabilities of the words in the given sentence.
   * @param doc a sentence.
   * @return an array of 
   */
  getProbsSync : function(doc) {
    var words = this.getFeatures(doc); // an array with the unique words in the text, for example: [ 'free', 'watches' ]
    var cats = this.getCats(); // a hash with the possible categories: { 'cat1': 1, 'cat2': 1 }
    var counts = this.getWordCounts(words, cats); // For each word encountered during training, the counts of times it occured in each category. 
    var probs = this.getCatProbs(cats, words, counts); // The probabilities that the given document belongs to each of the categories, i.e.: { 'cat1': 0.1875, 'cat2': 0.0625 }
    return probs;
  },

  bestMatch : function(probs) {
    var max = _(probs).reduce(function(max, prob, cat) {
      return max.prob > prob ? max : {cat: cat, prob: prob};
    }, {prob: 0});

    var category = max.cat || this.default;
    var threshold = this.thresholds[max.cat] || 1;

    _(probs).map(function(prob, cat) {
     if (!(cat == max.cat) && prob * threshold > max.prob) {
       category = this.default; // not greater than other category by enough
     }
    }, this);

    return category;
  },

  /**
   * Ask the classifier what category the given document belongs to.
   * @param doc [string] a sentence.
   * @return the most probable class of this sample.
   */
  classify : function(doc, callback) {
    if (!this.backend.async) {
      return this.classifySync(doc);
    }

    var that = this;
    this.getProbs(doc, function(probs) {
      callback(that.bestMatch(probs));
    });
  },

  /**
   * Ask the classifier what category the given document belongs to.
   * @param doc [string] a sentence.
   * @return the most probable class of this sample.
   */
  classifySync : function(doc) {
    var probs = this.getProbsSync(doc);
    return this.bestMatch(probs);
  },

  /**
   * Test the classifier on the given documents.
   * @param data an array with objects of the format: {input: sample1, output: class1}
   */
  test : function(data) {
    // number of classification errors:
    var error = 0;
    data.forEach(function(datum) {
      var output = this.classify(datum.input);
      error += output == datum.output ? 0 : 1;
    }, this);
    return error / data.length;
  },

  toJSON : function(callback) {
    return this.backend.toJSON(callback);
  },

  fromJSON : function(json, callback) {
    this.backend.fromJSON(json, callback);
    return this;
  }
}

//EREL: consistency with other classifiers:
Bayesian.prototype.trainBatch  = Bayesian.prototype.trainAll;
Bayesian.prototype.trainOnline = Bayesian.prototype.train;

exports.Bayesian = Bayesian;
