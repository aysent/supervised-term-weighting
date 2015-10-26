# Supervised Term Weighting Schemes for Text Classification

This repository accompanies the following blog post:

http://aysent.github.io/2015/10/21/supervised-term-weighting.html

# Getting started

1. Clone this repository:

    git clone https://github.com/aysent/supervised-term-weighting
    cd supervised-term-weighting

2. Download and unpack the Large Movie Review Dataset v1.0 from:

http://ai.stanford.edu/~amaas/data/sentiment/ [1]

3. [Optional] Files with filenames of training and test samples (train.csv and test.csv) are created with:

    source create_filelists.sh

4. Transform reviews with unsupervised (tf-idf) or supervised term weighting schemes and compare their performance:

    python train.py


[1] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. *The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)*.

