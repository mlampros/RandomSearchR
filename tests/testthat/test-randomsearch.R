#==================================================================================================================================================================
# data sets

# regression
data(Boston, package = 'MASS')
X = Boston[, -dim(Boston)[2]]
y1 = Boston[, dim(Boston)[2]]
form <- as.formula(paste('medv ~', paste(names(X), collapse = '+')))
ALL_DATA = Boston

# binary classification
data(ionosphere, package = 'kknn')
ionosphere = ionosphere[, -2]                                                                             # remove second column which has a single unique value
X_class = ionosphere[, -dim(ionosphere)[2]]
y1_class = ionosphere[, dim(ionosphere)[2]]
form_class <- as.formula(paste('class ~', paste(names(X_class), collapse = '+')))
y1_class = c(1:length(unique(y1_class)))[ match(ionosphere$class, sort(unique(ionosphere$class))) ]       # labels should begin from 1:Inf
ALL_DATA_class = ionosphere

all_data_gbm = ionosphere
all_data_gbm$class = y1_class -1           # gbm exception

ALL_DATA_class$class = as.factor(y1_class)

nnet_dat = ALL_DATA_class
nnet_dat$Y = nnet::class.ind(nnet_dat$class)    # nnet exception
nnet_dat$class = NULL
form_class_nnet <- as.formula(paste('Y ~', paste(names(X_class), collapse = '+')))

# multiclass classification
data(glass, package = 'kknn')
X_mlt = glass[, -c(1, dim(glass)[2])]
y1_mlt = glass[, dim(glass)[2]]
form_mlt <- as.formula(paste('Type ~', paste(names(X_mlt),collapse = '+')))
y1_mlt = c(1:length(unique(y1_mlt)))[ match(y1_mlt, sort(unique(y1_mlt))) ]                               # labels should begin from 1:Inf
ALL_DATA_mlt = glass
ALL_DATA_mlt$Type = as.factor(y1_mlt)

nnet_dat_mlt = ALL_DATA_mlt
nnet_dat_mlt$Y = nnet::class.ind(nnet_dat_mlt$Type)  # nnet exception
nnet_dat_mlt$Type = NULL
form_mlt_nnet <- as.formula(paste('Y ~', paste(names(X_mlt),collapse = '+')))


# data for elmNNRcpp
yy = matrix(y1, nrow = length(y1), ncol = 1)
yy_class = elmNNRcpp::onehot_encode(y1_class - 1)
yyy_class = elmNNRcpp::onehot_encode(y1_mlt - 1)

#====================================================================================================================================================================

# function to validate the parameters of each algorithm

func_validate = function(RES, dat, tune_iters, repeats_or_folds, method, each_resampling_proportion = 2/3, names_grid_params, regression) {          # example each_resampling_proportion : in case of 3-fold cross_validation is the proportion of train-data for each fold

  PREDS = RES$PREDS
  PARAMS = RES$PARAMS

  length_sublists = mean(unlist(lapply(PREDS, length))) == repeats_or_folds                                               # should be equal to repeats

  length_nested_sublists = mean(unlist(lapply(PREDS, function(x) unlist(lapply(x, length))))) == 4            # should be equal to 4 : pred_tr, pred_te, y_tr, y_te

  if (method == 'bootstrap') {     # in case of bootstrap draws are with replacement, thus same observations can appear multiple times in both train-test [ check length only of train splits ]

    if (regression) {

      is_length_of_data_correct = all.equal(mean(unlist(lapply(PREDS, function(x) unlist(lapply(x, function(y) length(y$pred_tr)))))), nrow(dat) * each_resampling_proportion, tolerance = 0.05)}     # it tests if value is approximately correct [ not exact correct using a tolerance rate ]. relevant especially in cross_validation as the splits can be unequal

    else {

      is_length_of_data_correct = all.equal(mean(unlist(lapply(PREDS, function(x) unlist(lapply(x, function(y) nrow(y$pred_tr)))))), nrow(dat) * each_resampling_proportion, tolerance = 0.05)
    }

    is_length_of_response_data_correct = all.equal(mean(unlist(lapply(PREDS, function(x) unlist(lapply(x, function(y) length(y$y_tr)))))), nrow(dat) * each_resampling_proportion, tolerance = 0.05)
  }

  else {

    if (regression) {

      is_length_of_data_correct = mean(unlist(lapply(PREDS, function(x) unlist(lapply(x, function(y) length(y$pred_tr) + length(y$pred_te)))))) == nrow(dat)}

    else {

      is_length_of_data_correct = mean(unlist(lapply(PREDS, function(x) unlist(lapply(x, function(y) nrow(y$pred_tr) + nrow(y$pred_te)))))) == nrow(dat)
    }

    is_length_of_response_data_correct = mean(unlist(lapply(PREDS, function(x) unlist(lapply(x, function(y) length(y$y_tr) + length(y$y_te)))))) == nrow(dat)
  }

  is_length_params_correct = mean(apply(PARAMS, 2, length)) == tune_iters

  is_names_grid_params_correct = sum(colnames(PARAMS) %in% names_grid_params) == length(names_grid_params)

  return(list(length_sublists, length_nested_sublists, is_length_of_data_correct, is_length_of_response_data_correct,

              is_length_params_correct, is_names_grid_params_correct))
}

#=============================================================================================================================================================================

context("Randomsearch testing")


#=================
# Error handling
#================

testthat::test_that("if the resampling method is null it returns an error", {

  grid_car = list(k = 3:20)

  testthat::expect_error(random_search_resample(y1, tune_iters = 5,

                                                resampling_method = NULL,

                                                ALGORITHM = list(package = require(caret), algorithm = knnreg),

                                                grid_params = grid_car,

                                                DATA = list(x = X, y = y1),

                                                Args = NULL,

                                                regression = TRUE, re_run_params = FALSE))
})


testthat::test_that("if the resampling_method$repeats AND resampling_method$sample_rate AND resampling_method$folds is null it returns an error", {

  grid_car = list(k = 3:20)

  testthat::expect_error(random_search_resample(y1, tune_iters = 5,

                                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = NULL),

                                                ALGORITHM = list(package = require(caret), algorithm = knnreg),

                                                grid_params = grid_car,

                                                DATA = list(x = X, y = y1),

                                                Args = NULL,

                                                regression = TRUE, re_run_params = FALSE))
})



testthat::test_that("if the resampling_method$repeats < 2 AND resampling_method$sample_rate > 0 AND resampling_method$folds is null, it returns an error", {

  grid_car = list(k = 3:20)

  testthat::expect_error(random_search_resample(y1, tune_iters = 5,

                                                resampling_method = list(method = 'cross_validation', repeats = 1, sample_rate = 0.75, folds = NULL),

                                                ALGORITHM = list(package = require(caret), algorithm = knnreg),

                                                grid_params = grid_car,

                                                DATA = list(x = X, y = y1),

                                                Args = NULL,

                                                regression = TRUE, re_run_params = FALSE))
})



testthat::test_that("if the resampling_method$repeats = NULL AND resampling_method$sample_rate = NULL AND resampling_method$folds < 2, it returns an error", {

  grid_car = list(k = 3:20)

  testthat::expect_error(random_search_resample(y1, tune_iters = 5,

                                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 1),

                                                ALGORITHM = list(package = require(caret), algorithm = knnreg),

                                                grid_params = grid_car,

                                                DATA = list(x = X, y = y1),

                                                Args = NULL,

                                                regression = TRUE, re_run_params = FALSE))
})


testthat::test_that("if the tune_iters = NULL, it returns an error", {

  grid_car = list(k = 3:20)

  testthat::expect_error(random_search_resample(y1, tune_iters = NULL,

                                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                                ALGORITHM = list(package = require(caret), algorithm = knnreg),

                                                grid_params = grid_car,

                                                DATA = list(x = X, y = y1),

                                                Args = NULL,

                                                regression = TRUE, re_run_params = FALSE))
})


testthat::test_that("if the tune_iters < 1, it returns an error", {

  grid_car = list(k = 3:20)

  testthat::expect_error(random_search_resample(y1, tune_iters = 0,

                                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                                ALGORITHM = list(package = require(caret), algorithm = knnreg),

                                                grid_params = grid_car,

                                                DATA = list(x = X, y = y1),

                                                Args = NULL,

                                                regression = TRUE, re_run_params = FALSE))
})



testthat::test_that("if the resampling_method$method is either 'bootstrap' or 'train_test_split' and the resampling_method$folds is not NULL, it returns an error", {

  grid_car = list(k = 3:20)

  testthat::expect_error(random_search_resample(y1, tune_iters = 1,

                                                resampling_method = list(method = 'bootstrap', repeats = NULL, sample_rate = NULL, folds = 3),

                                                ALGORITHM = list(package = require(caret), algorithm = knnreg),

                                                grid_params = grid_car,

                                                DATA = list(x = X, y = y1),

                                                Args = NULL,

                                                regression = TRUE, re_run_params = FALSE))
})



testthat::test_that("if the resampling_method$method is either 'bootstrap' or 'train_test_split' and the resampling_method$folds is not NULL, it returns an error", {

  grid_car = list(k = 3:20)

  testthat::expect_error(random_search_resample(y1, tune_iters = 1,

                                                resampling_method = list(method = 'cross_validation', repeats = 2, sample_rate = 0.75, folds = 3),

                                                ALGORITHM = list(package = require(caret), algorithm = knnreg),

                                                grid_params = grid_car,

                                                DATA = list(x = X, y = y1),

                                                Args = NULL,

                                                regression = TRUE, re_run_params = FALSE))
})


#====================
# testing algorithms
#====================


#=================================================================================================================================================================================

# Regression elmNNRcpp
#---------------------

testthat::test_that("results for elmNNRcpp using the func_validate function are correct, REGRESSION", {

  grid = list(nhid = seq(100, 200, 10), actfun = c('sig', 'purelin', 'relu'), init_weights = c('normal_gaussian', 'uniform_positive', 'uniform_negative' ),

              bias = c(T,F), leaky_relu_alpha = c(0.0, 0.01, 0.05, 0.1) )

  algs = random_search_resample(yy, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(elmNNRcpp), algorithm = elm_train),

                                grid_params = grid,

                                DATA = list(y = yy, x = as.matrix(X)),

                                Args = NULL,

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})



# binary Classification elmNNRcpp
#--------------------------------

testthat::test_that("results for elmNNRcpp using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(nhid = seq(100, 200, 100), actfun = c('sig', 'sin', 'radbas', 'hardlim', 'hardlims', 'satlins', 'tansig', 'tribas', 'relu', 'purelin' ),

              init_weights = c('normal_gaussian', 'uniform_positive', 'uniform_negative' ),

              bias = c(T,F), leaky_relu_alpha = c(0.0, 0.01, 0.05, 0.1) )


  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(elmNNRcpp), algorithm = elm_train),

                                grid_params = grid,

                                DATA = list(y = yy_class, x = as.matrix(scale(X_class))),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 4)
})



# multiclass Classification elmNNRcpp
#------------------------------------

testthat::test_that("results for elmNNRcpp using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(nhid = seq(100, 200, 10), actfun = c('sig', 'sin', 'radbas', 'hardlim', 'hardlims', 'satlins', 'tansig', 'tribas', 'relu', 'purelin' ),

              init_weights = c('normal_gaussian', 'uniform_positive', 'uniform_negative' ),

              bias = c(T,F), leaky_relu_alpha = c(0.0, 0.01, 0.05, 0.1) )


  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(elmNNRcpp), algorithm = elm_train),

                                grid_params = grid,

                                DATA = list(y = yyy_class, x = as.matrix(scale(X_mlt))),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 4)
})


#=================================================================================================================================================================================

# randomForest for regression

testthat::test_that("results for randomForest using the func_validate function are correct, REGRESSION", {

  grid = list(ntree = 5:20, mtry = c(2:4))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(randomForest), algorithm = randomForest),

                                grid_params = grid,

                                DATA = list(y = y1, x = X),

                                Args = NULL,

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for randomForest using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(ntree = 5:20, mtry = c(2:4))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(randomForest), algorithm = randomForest),

                                grid_params = grid,

                                DATA = list(y = as.factor(y1_class), x = X_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for randomForest using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(ntree = 5:20, mtry = c(2:4))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(randomForest), algorithm = randomForest),

                                grid_params = grid,

                                DATA = list(y = as.factor(y1_mlt), x = X_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================

# ksvm for regression

testthat::test_that("results for ksvm using the func_validate function are correct, REGRESSION", {

  grid = list(type = c('eps-svr', 'nu-svr'), C = c(0.1, 0.5, 1, 2, 10, 100), nu = c(0.1, 0.2, 0.5))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(kernlab), algorithm = ksvm),

                                grid_params = grid,

                                DATA = list(y = y1, x = as.matrix(X)),

                                Args = NULL,

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for ksvm using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(type = c('C-svc', 'C-bsvc'), C = c(0.1, 0.5, 1, 2, 10, 100), nu = c(0.1, 0.2, 0.5))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(kernlab), algorithm = ksvm),

                                grid_params = grid,

                                DATA = list(y = as.factor(y1_class), x = as.matrix(X_class)),

                                Args = list(prob.model = TRUE, scaled = FALSE),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for ksvm using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(type = c('C-svc', 'C-bsvc'), C = c(0.1, 0.5, 1, 2, 10, 100), nu = c(0.1, 0.2, 0.5))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(kernlab), algorithm = ksvm),

                                grid_params = grid,

                                DATA = list(y = as.factor(y1_mlt), x = as.matrix(X_mlt)),

                                Args = list(prob.model = TRUE, scaled = FALSE),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================


# kknn for regression

testthat::test_that("results for kknn using the func_validate function are correct, REGRESSION", {

  grid = list(k = 3:20, distance = c(1:5), kernel = c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal"))

  algs = random_search_resample(y1, tune_iters = 3,

                                    resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                    ALGORITHM = list(package = require(kknn), algorithm = kknn),

                                    grid_params = grid,

                                    DATA = list(formula = form, train = ALL_DATA),

                                    Args = NULL,

                                    regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for kknn using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(k = 3:20, distance = c(1:5), kernel = c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal"))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(kknn), algorithm = kknn),

                                grid_params = grid,

                                DATA = list(formula = form_class, train = ALL_DATA_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for kknn using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(k = 3:20, distance = c(1:5), kernel = c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal"))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(kknn), algorithm = kknn),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, train = ALL_DATA_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#=================================================================================================================================================================================

# caret knnreg, knn

testthat::test_that("results for caret-knn using the func_validate function are correct, REGRESSION", {

  grid = list(k = 3:20)

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(caret), algorithm = knnreg),

                                grid_params = grid,

                                DATA = list(x = X, y = y1),

                                Args = NULL,

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for caret-knn3 using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(k = 3:20)

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(caret), algorithm = knn3),

                                grid_params = grid,

                                DATA = list(x = X_class, y = as.factor(y1_class)),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for caret-knn using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(k = 3:20)

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(caret), algorithm = knn3),

                                grid_params = grid,

                                DATA = list(x = X_mlt, y = as.factor(y1_mlt)),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================

# RWeka knn

testthat::test_that("results for RWeka knn using the func_validate function are correct, REGRESSION", {

  grid = list(control = RWeka::Weka_control(K = seq(3, 20, 1), I = c(TRUE, FALSE), X = c(TRUE, FALSE)))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(RWeka), algorithm = IBk),

                                grid_params = grid,

                                DATA = list(formula = form, data = ALL_DATA),

                                Args = NULL,

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid$control), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for RWeka knn using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(K = seq(3, 20, 1), I = c(TRUE, FALSE), X = c(TRUE, FALSE)))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = IBk),

                                grid_params = grid,

                                DATA = list(formula = form_class, data = ALL_DATA_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for RWeka knn using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(K = seq(3, 20, 1), I = c(TRUE, FALSE), X = c(TRUE, FALSE)))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = IBk),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, data = ALL_DATA_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#=================================================================================================================================================================================

# RWeka AdaBoostM1 [ defaults to DecisionStump , difficult to add further base classifiers such as J48, as it requires an additional list of arguments ]
# Classification ONLY

# binary classification

testthat::test_that("results for RWeka AdaBoostM1 using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(P = c(70, 80, 90), I = seq(5, 10, 1)))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = AdaBoostM1),

                                grid_params = grid,

                                DATA = list(formula = form_class, data = ALL_DATA_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for RWeka AdaBoostM1 using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(P = c(70, 80, 90), I = seq(5, 10, 1)))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = AdaBoostM1),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, data = ALL_DATA_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#==================================================================================================================================================

# RWeka Bagging

testthat::test_that("results for RWeka Bagging using the func_validate function are correct, REGRESSION", {

  grid = list(control = RWeka::Weka_control(P = c(70, 80, 90), I = seq(5, 10, 1)))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(RWeka), algorithm = Bagging),

                                grid_params = grid,

                                DATA = list(formula = form, data = ALL_DATA),

                                Args = NULL,

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid$control), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for RWeka Bagging using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(P = c(70, 80, 90), I = seq(5, 10, 1)))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = Bagging),

                                grid_params = grid,

                                DATA = list(formula = form_class, data = ALL_DATA_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for RWeka Bagging using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(P = c(70, 80, 90), I = seq(5, 10, 1)))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = Bagging),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, data = ALL_DATA_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#==================================================================================================================================================

# RWeka LogitBoost [ Classification ONLY ]

# binary classification

testthat::test_that("results for RWeka LogitBoost using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(P = c(70, 80, 90), I = seq(5, 10, 1), H = c(0.1, 0.5, 1.0)))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = LogitBoost),

                                grid_params = grid,

                                DATA = list(formula = form_class, data = ALL_DATA_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for RWeka LogitBoost using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(P = c(70, 80, 90), I = seq(5, 10, 1), H = c(0.1, 0.5, 1.0)))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = LogitBoost),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, data = ALL_DATA_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#==================================================================================================================================================

# RWeka J48  [ Classification ONLY ]


# binary classification

testthat::test_that("results for RWeka J48 using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(M = c(2, 5, 10), R = c(TRUE, FALSE), B = c(TRUE, FALSE),

                                            S = c(TRUE, FALSE), A = c(TRUE, FALSE), J = c(TRUE, FALSE)))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = J48),

                                grid_params = grid,

                                DATA = list(formula = form_class, data = ALL_DATA_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for RWeka J48 using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control(M = c(2, 5, 10), R = c(TRUE, FALSE), B = c(TRUE, FALSE), S = c(TRUE, FALSE),

                                            A = c(TRUE, FALSE), J = c(TRUE, FALSE)))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = J48),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, data = ALL_DATA_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#=================================================================================================================================================================================


# RWeka M5P  [ Regression ONLY ]

testthat::test_that("results for RWeka M5P using the func_validate function are correct, REGRESSION", {

  grid = list(control = RWeka::Weka_control(M = c(2, 4, 10), N = c(TRUE, FALSE), U = c(TRUE, FALSE), R = c(TRUE, FALSE)))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = M5P),

                                grid_params = grid,

                                DATA = list(formula = form, data = ALL_DATA),

                                Args = NULL,

                                regression = T, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid$control), T))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================

# RWeka M5Rules  [ Regression ONLY ]

testthat::test_that("results for RWeka M5P using the func_validate function are correct, REGRESSION", {

  grid = list(control = RWeka::Weka_control(M = c(2, 4, 10), N = c(TRUE, FALSE), U = c(TRUE, FALSE), R = c(TRUE, FALSE)))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = M5Rules),

                                grid_params = grid,

                                DATA = list(formula = form, data = ALL_DATA),

                                Args = NULL,

                                regression = T, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid$control), T))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================

# RWeka SMO  [ Classification ONLY ]


# binary classification

testthat::test_that("results for RWeka SMO using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control( C = c(0.5, 1.0, 5), N = c(0, 1, 2), M = c(TRUE, FALSE)))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = SMO),

                                grid_params = grid,

                                DATA = list(formula = form_class, data = ALL_DATA_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for RWeka SMO using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(control = RWeka::Weka_control( C = c(0.5, 1.0, 5), N = c(0, 1, 2), M = c(TRUE, FALSE)))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(RWeka), algorithm = SMO),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, data = ALL_DATA_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid$control), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#=================================================================================================================================================================================

# gbm

testthat::test_that("results for gbm using the func_validate function are correct, REGRESSION", {

  grid = list(n.trees = seq(5, 15, 1), shrinkage = c(0.01, 0.1, 0.5))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(gbm), algorithm = gbm),

                                grid_params = grid,

                                DATA = list(formula = form, data = ALL_DATA),

                                Args = list(distribution = 'gaussian'),

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for gbm using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(n.trees = seq(5, 15, 1), shrinkage = c(0.01, 0.1, 0.5))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(gbm), algorithm = gbm),

                                grid_params = grid,

                                DATA = list(formula = form_class, data = all_data_gbm),         # exception, as it requires the response to be in c(0,1)

                                Args = list(distribution = 'bernoulli'),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for gbm using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(n.trees = seq(5, 15, 1), shrinkage = c(0.01, 0.1, 0.5))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(gbm), algorithm = gbm),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, data = ALL_DATA_mlt),

                                Args = list(distribution = 'multinomial'),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#=================================================================================================================================================================================

# h2o.randomForest

testthat::test_that("results for h2o using the func_validate function are correct, REGRESSION", {

  grid = list(ntrees = 5:15, mtries = c(1:3))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(h2o), algorithm = h2o.randomForest),

                                grid_params = grid,

                                DATA = list(h2o.x = X, h2o.y = y1),

                                Args = NULL,

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for h2o using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(ntrees = 5:15, mtries = c(1:3))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(h2o), algorithm = h2o.randomForest),

                                grid_params = grid,

                                DATA = list(h2o.x = X_class, h2o.y = as.factor(y1_class)),          # response should be a factor

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for h2o using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(ntrees = 5:15, mtries = c(1:3))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(h2o), algorithm = h2o.randomForest),

                                grid_params = grid,

                                DATA = list(h2o.x = X_mlt, h2o.y = as.factor(y1_mlt)),          # response should be a factor

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================

# h2o.deeplearning        [ it works if the hidden units appear in Args ]

testthat::test_that("results for h2o using the func_validate function are correct, REGRESSION", {

  grid = list(activation = c("Rectifier", "Tanh"), epochs = seq(5,10,1))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(h2o), algorithm = h2o.deeplearning),

                                grid_params = grid,

                                DATA = list(h2o.x = X, h2o.y = y1),

                                Args = list(distribution = 'gaussian', hidden = c(10, 10)),

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for h2o using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(activation = c("Rectifier", "Tanh"), epochs = seq(5,10,1))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(h2o), algorithm = h2o.deeplearning),

                                grid_params = grid,

                                DATA = list(h2o.x = X_class, h2o.y = as.factor(y1_class)),          # response should be a factor

                                Args = list(distribution = 'bernoulli', hidden = c(10, 10)),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for h2o using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(activation = c("Rectifier", "Tanh"), epochs = seq(5,10,1))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(h2o), algorithm = h2o.deeplearning),

                                grid_params = grid,

                                DATA = list(h2o.x = X_mlt, h2o.y = as.factor(y1_mlt)),          # response should be a factor

                                Args = list(distribution = 'multinomial', hidden = c(10, 10)),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#=================================================================================================================================================================================

# h2o.gbm

testthat::test_that("results for h2o.gbm using the func_validate function are correct, REGRESSION", {

  grid = list(ntrees = seq(5, 10, 1), max_depth = seq(5,10,1))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(h2o), algorithm = h2o.gbm),

                                grid_params = grid,

                                DATA = list(h2o.x = X, h2o.y = y1),

                                Args = list(distribution = 'gaussian'),

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for h2o.gbm using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(ntrees = seq(5, 10, 1), max_depth = seq(5,10,1))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(h2o), algorithm = h2o.gbm),

                                grid_params = grid,

                                DATA = list(h2o.x = X_class, h2o.y = as.factor(y1_class)),          # response should be a factor

                                Args = list(distribution = 'bernoulli'),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for h2o.gbm using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(ntrees = seq(5, 10, 1), max_depth = seq(5,10,1))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(h2o), algorithm = h2o.gbm),

                                grid_params = grid,

                                DATA = list(h2o.x = X_mlt, h2o.y = as.factor(y1_mlt)),          # response should be a factor

                                Args = list(distribution = 'multinomial'),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#=================================================================================================================================================================================

# xgboost   [ BEWARE : IN BINARY AND MULTICLASS CLASSIFICATION THE LABELS SHOULD BEGIN FROM 1:Inf, AS I SUBTRACT INTERNALLY 1.0 ]


testthat::test_that("results for xgboost using the func_validate function are correct, REGRESSION", {

  grid = list(params = list("objective" = "reg:linear", "bst:eta" = seq(0.05, 0.1, 0.005), "subsample" = seq(0.65, 0.85, 0.05),

                            "max_depth" = seq(3, 5, 1), "eval_metric" = "rmse", "colsample_bytree" = seq(0.65, 0.85, 0.05),

                            "lambda" = 1e-5, "alpha" = 1e-5, "nthread" = 4))


  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = library(xgboost), algorithm = xgb.train),

                                grid_params = grid,

                                DATA = list(watchlist = list(label = y1, data = X)),

                                Args = list(nrounds = 5, verbose = 0, print.every.n = 10, early.stop.round = 5, maximize = FALSE),

                                regression = TRUE, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid$params), T))

  testthat::expect_true(sum(valid_all) == 6)
})

# binary classification

testthat::test_that("results for xgboost using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(params = list("objective" = "binary:logistic", "bst:eta" = seq(0.05, 0.1, 0.005), "subsample" = seq(0.65, 0.85, 0.05),

                            "max_depth" = seq(3, 5, 1), "eval_metric" = "error", "colsample_bytree" = seq(0.65, 0.85, 0.05),

                            "lambda" = 1e-5, "alpha" = 1e-5, "nthread" = 4))


  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = library(xgboost), algorithm = xgb.train),

                                grid_params = grid,

                                DATA = list(watchlist = list(label = y1_class, data = X_class)),

                                Args = list(nrounds = 5, verbose = 0, print.every.n = 10, early.stop.round = 5, maximize = FALSE),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid$params), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for xgboost using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(params = list("objective" = "multi:softprob", "num_class" = 6, "bst:eta" = seq(0.05, 0.1, 0.005), "subsample" = seq(0.65, 0.85, 0.05),

                            "max_depth" = seq(3, 5, 1), "eval_metric" = "mlogloss", "colsample_bytree" = seq(0.65, 0.85, 0.05),

                            "lambda" = 1e-5, "alpha" = 1e-5, "nthread" = 4))


  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = library(xgboost), algorithm = xgb.train),

                                grid_params = grid,

                                DATA = list(watchlist = list(label = y1_mlt, data = X_mlt)),

                                Args = list(nrounds = 5, verbose = 0, print.every.n = 10, early.stop.round = 5, maximize = FALSE),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid$params), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#=================================================================================================================================================================================

# svm  [ e1071 package ]

# regression

testthat::test_that("results for svm using the func_validate function are correct, REGRESSION", {

  grid = list(degree = 3:5, gamma = c(0.01, 0.05, 0.1), cost = c(0.1, 0.5, 1.0, 5.0))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(e1071), algorithm = svm),

                                grid_params = grid,

                                DATA = list(formula = form, data = ALL_DATA),

                                Args = list(kernel = 'radial'),

                                regression = T, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})


# binary classification

testthat::test_that("results for svm using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(degree = 3:5, gamma = c(0.01, 0.05, 0.1), cost = c(0.1, 0.5, 1.0, 5.0))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(e1071), algorithm = svm),

                                grid_params = grid,

                                DATA = list(formula = form_class, data = ALL_DATA_class),

                                Args = list(kernel = 'radial', probability = T),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for svm using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(degree = 3:5, gamma = c(0.01, 0.05, 0.1), cost = c(0.1, 0.5, 1.0, 5.0))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(e1071), algorithm = svm),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, data = ALL_DATA_mlt),

                                Args = list(kernel = 'radial', probability = T),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================

# LiblineaR

# regression

testthat::test_that("results for LiblineaR using the func_validate function are correct, REGRESSION", {

  grid = list(type = c(11,12,13), bias = c(T,F), cost = c(0.1, 0.5, 1.0, 5.0), epsilon = c(0.01, 0.05, 0.1))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(LiblineaR), algorithm = LiblineaR),

                                grid_params = grid,

                                DATA = list(target = y1, data = X),

                                Args = NULL,

                                regression = T, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})


# binary classification

testthat::test_that("results for LiblineaR using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(type = c(0,6,7), bias = c(T,F), cost = c(0.1, 0.5, 1.0, 5.0), epsilon = c(0.01, 0.05, 0.1))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(LiblineaR), algorithm = LiblineaR),

                                grid_params = grid,

                                DATA = list(target = y1_class, data = X_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for LiblineaR using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(type = c(0,6,7), bias = c(T,F), cost = c(0.1, 0.5, 1.0, 5.0), epsilon = c(0.01, 0.05, 0.1))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(LiblineaR), algorithm = LiblineaR),

                                grid_params = grid,

                                DATA = list(target = y1_mlt, data = X_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================


# extraTrees

# regression

testthat::test_that("results for extraTrees using the func_validate function are correct, REGRESSION", {

  grid = list(ntree = 5:10, mtry = 2:4, quantile = c(T,F))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(extraTrees), algorithm = extraTrees),

                                grid_params = grid,

                                DATA = list(y = y1, x = X),

                                Args = NULL,

                                regression = T, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})


# binary classification

testthat::test_that("results for extraTrees using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(ntree = 5:10, mtry = 2:4)

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(extraTrees), algorithm = extraTrees),

                                grid_params = grid,

                                DATA = list(y = as.factor(y1_class), x = X_class),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for extraTrees using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(ntree = 5:10, mtry = 2:4)

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(extraTrees), algorithm = extraTrees),

                                grid_params = grid,

                                DATA = list(y = as.factor(y1_mlt), x = X_mlt),

                                Args = NULL,

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================


# glmnet  [ here cv.glmnet will be used ]

# regression

testthat::test_that("results for cv.glmnet using the func_validate function are correct, REGRESSION", {

  grid = list(alpha = c(0, 0.5, 1.0), nlambda = 50:100, standardize = c(T,F))

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(glmnet), algorithm = cv.glmnet),

                                grid_params = grid,

                                DATA = list(y = y1, x = as.matrix(X)),

                                Args = list(family = 'gaussian', type.measure = 'mse', nfolds = 3, parallel = F),

                                regression = T, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})


# binary classification

testthat::test_that("results for cv.glmnet using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(alpha = c(0, 0.5, 1.0), nlambda = 50:100, standardize = c(T,F))

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(glmnet), algorithm = cv.glmnet),

                                grid_params = grid,

                                DATA = list(y = as.factor(y1_class), x = as.matrix(X_class)),

                                Args = list(family = 'binomial', type.measure = 'class', nfolds = 3, parallel = F),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for cv.glmnet using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(alpha = c(0, 0.5, 1.0), nlambda = 50:100, standardize = c(T,F))

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(glmnet), algorithm = cv.glmnet),

                                grid_params = grid,

                                DATA = list(y = as.factor(y1_mlt), x = as.matrix(X_mlt)),

                                Args = list(family = 'multinomial', type.measure = 'class', nfolds = 3, parallel = F),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================

# nnet

# regression

testthat::test_that("results for nnet using the func_validate function are correct, REGRESSION", {

  grid = list(size = 15:20, maxit = 100:265)

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(nnet), algorithm = nnet),

                                grid_params = grid,

                                DATA = list(y = y1, x = X),

                                Args = list(linout = T, trace = F),

                                regression = T, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})


# binary classification

testthat::test_that("results for nnet using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(size = 15:20, maxit = 100:265)

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(nnet), algorithm = nnet),

                                grid_params = grid,

                                DATA = list(formula = form_class_nnet, data = nnet_dat),                # SPECIAL CASE FOR nnet [ http://stackoverflow.com/questions/19870594/using-softmax-in-nnet-r-for-target-column-with-more-than-2-states ]

                                Args = list(linout = F, trace = F, softmax = T, entropy = T),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for nnet using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(size = 15:20, maxit = 100:265)

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(nnet), algorithm = nnet),

                                grid_params = grid,

                                DATA = list(formula = form_mlt_nnet, data = nnet_dat_mlt),                # SPECIAL CASE FOR nnet [ http://stackoverflow.com/questions/19870594/using-softmax-in-nnet-r-for-target-column-with-more-than-2-states ]

                                Args = list(linout = F, trace = F, softmax = T, entropy = T),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


#=================================================================================================================================================================================

# ranger

# regression

testthat::test_that("results for ranger using the func_validate function are correct, REGRESSION", {

  grid = list(num.trees = 5:10, mtry = 2:4)

  algs = random_search_resample(y1, tune_iters = 3,

                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                                ALGORITHM = list(package = require(ranger), algorithm = ranger),

                                grid_params = grid,

                                DATA = list(formula = form, data = ALL_DATA),

                                Args = list(write.forest = T, probability = F, verbose = F),

                                regression = T, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA, 3, 3, 'cross_validation', each_resampling_proportion = 2/3, names(grid), T))

  testthat::expect_true(sum(valid_all) == 6)
})


# binary classification

testthat::test_that("results for ranger using the func_validate function are correct, BINARY CLASSIFICATION", {

  grid = list(num.trees = 5:10, mtry = 2:4)

  algs = random_search_resample(as.factor(y1_class), tune_iters = 3,

                                resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(ranger), algorithm = ranger),

                                grid_params = grid,

                                DATA = list(formula = form_class, data = ALL_DATA_class),

                                Args = list(write.forest = T, probability = T, verbose = F),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_class, 3, 5, 'train_test_split', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})


# multiclass classification

testthat::test_that("results for ranger using the func_validate function are correct, MULTICLASS CLASSIFICATION", {

  grid = list(num.trees = 5:10, mtry = 2:4)

  algs = random_search_resample(as.factor(y1_mlt), tune_iters = 3,

                                resampling_method = list(method = 'bootstrap', repeats = 3, sample_rate = 2/3, folds = NULL),

                                ALGORITHM = list(package = require(ranger), algorithm = ranger),

                                grid_params = grid,

                                DATA = list(formula = form_mlt, data = ALL_DATA_mlt),

                                Args = list(write.forest = T, probability = T, verbose = F),

                                regression = F, re_run_params = FALSE)

  valid_all = unlist(func_validate(algs, ALL_DATA_mlt, 3, 3, 'bootstrap', each_resampling_proportion = 2/3, names(grid), F))

  testthat::expect_true(sum(valid_all) == 6)
})

#=================================================================================================================================================================================

