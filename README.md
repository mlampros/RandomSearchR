[![Travis-CI Build Status](https://travis-ci.org/mlampros/RandomSearchR.svg?branch=master)](https://travis-ci.org/mlampros/RandomSearchR)
[![codecov.io](https://codecov.io/github/mlampros/RandomSearchR/coverage.svg?branch=master)](https://codecov.io/github/mlampros/RandomSearchR?branch=master)



### Random search in R

The functions in the package can be used to,
* find the optimal parameters of an algorithm using random search
* compare performance of algorithms using summary statistics. 

<br>
To install the package from Github use the 'install_github' function of the devtools package,
<br><br>

```R

devtools::install_github('mlampros/RandomSearchR')

```
<br>

After downloading the package use ? to read info about each function (i.e. ?random_search_resample). More details can be found in the [blog post](http://mlampros.github.io/2016/03/14/random_search_R/).
<br><br>

UPDATED : 23-05-2016 [ I added tests and code-coverage ]

UPDATED : 13-09-2016 [ I replaced the *resampling_methods* function with the *repeated_resampling* function ]

UPDATED : 23-12-2017 [ I modified the xgboost-predict function due to an error and I added the *UNLABELED_TEST_DATA* parameter in the *random_search_resample* function, so that I can receive predictions for unlabeled data when I do random search ]
