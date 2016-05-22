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


#===========================================================================================

# evaluation metrics

mse = function(y_true, y_pred) {

  out = mean((y_true - y_pred) ^ 2)

  out
}


acc = function(y_true, preds) {

  out = table(y_true, max.col(preds, ties.method = "random"))

  acc = sum(diag(out))/sum(out)

  acc
}

#=============================================================================================

# sample some algorithms which will be used for testing the performance_measures function

# REGRESSION

TUNE_ITERS = 3

grid = list(k = 3:20, distance = c(1:5), kernel = c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal"))

algs = random_search_resample(y1, tune_iters = TUNE_ITERS,

                              resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                              ALGORITHM = list(package = require(kknn), algorithm = kknn),

                              grid_params = grid,

                              DATA = list(formula = form, train = ALL_DATA),

                              Args = NULL,

                              regression = TRUE, re_run_params = FALSE)


grid1 = list(k = 3:20)

algs1 = random_search_resample(y1, tune_iters = TUNE_ITERS,

                              resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                              ALGORITHM = list(package = require(caret), algorithm = knnreg),

                              grid_params = grid1,

                              DATA = list(x = X, y = y1),

                              Args = NULL,

                              regression = TRUE, re_run_params = FALSE)


grid2 = list(control = RWeka::Weka_control(K = seq(3, 20, 1), I = c(TRUE, FALSE), X = c(TRUE, FALSE)))

algs2 = random_search_resample(y1, tune_iters = TUNE_ITERS,

                              resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                              ALGORITHM = list(package = require(RWeka), algorithm = IBk),

                              grid_params = grid2,

                              DATA = list(formula = form, data = ALL_DATA),

                              Args = NULL,

                              regression = TRUE, re_run_params = FALSE)



# CLASSIFICATION

grid3 = list(control = RWeka::Weka_control(P = c(70, 80, 90), I = seq(5, 10, 1)))

algs3 = random_search_resample(as.factor(y1_class), tune_iters = TUNE_ITERS,

                              resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                              ALGORITHM = list(package = require(RWeka), algorithm = AdaBoostM1),

                              grid_params = grid3,

                              DATA = list(formula = form_class, data = ALL_DATA_class),

                              Args = NULL,

                              regression = F, re_run_params = FALSE)


grid4 = list(n.trees = seq(5, 15, 1), shrinkage = c(0.01, 0.1, 0.5))

algs4 = random_search_resample(as.factor(y1_class), tune_iters = TUNE_ITERS,

                              resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                              ALGORITHM = list(package = require(gbm), algorithm = gbm),

                              grid_params = grid4,

                              DATA = list(formula = form_class, data = all_data_gbm),         # exception, as it requires the response to be in c(0,1)

                              Args = list(distribution = 'bernoulli'),

                              regression = F, re_run_params = FALSE)


grid5 = list(activation = c("Rectifier", "Tanh"), epochs = seq(5,10,1))

algs5 = random_search_resample(as.factor(y1_class), tune_iters = TUNE_ITERS,

                              resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                              ALGORITHM = list(package = require(h2o), algorithm = h2o.deeplearning),

                              grid_params = grid5,

                              DATA = list(h2o.x = X_class, h2o.y = as.factor(y1_class)),          # response should be a factor

                              Args = list(distribution = 'bernoulli', hidden = c(10, 10)),

                              regression = F, re_run_params = FALSE)


grid6 = list(params = list("objective" = "binary:logistic", "bst:eta" = seq(0.05, 0.1, 0.005), "subsample" = seq(0.65, 0.85, 0.05),

                          "max_depth" = seq(3, 5, 1), "eval_metric" = "error", "colsample_bytree" = seq(0.65, 0.85, 0.05),

                          "lambda" = 1e-5, "alpha" = 1e-5, "nthread" = 4))

algs6 = random_search_resample(as.factor(y1_class), tune_iters = TUNE_ITERS,

                              resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                              ALGORITHM = list(package = library(xgboost), algorithm = xgb.train),

                              grid_params = grid6,

                              DATA = list(watchlist = list(label = y1_class, data = X_class)),

                              Args = list(nrounds = 5, verbose = 0, print.every.n = 10, early.stop.round = 5, maximize = FALSE),

                              regression = F, re_run_params = FALSE)

#=============================================================================================


context("Performance measures")


testthat::test_that("if the list object is NULL/empty it returns an error", {

  testthat::expect_error(performance_measures(NULL, mse, sort = list(variable = 'Median', decreasing = TRUE)))
})


testthat::test_that("in REGRESSION the performance_measures function returns : a list, the length of the list is 5, the names of the list are correct,

                    the rows of the 1st and 2nd sublists equals the number of tune_iters, the rows of the 3rd and 4th sublist equals the number of the algorithms used,

                    and the length of the last sublist equals the number of algorithms used", {

                    lst_obj = list(algs = algs, algs1 = algs1, algs2 = algs2)

                    perf = performance_measures(list_objects = lst_obj, eval_metric = mse, sort = list(variable = 'Median', decreasing = TRUE))

                    res_out = unlist(list(is.list(perf), length(perf) == 5, sum(names(perf) %in% c("train_params", "test_params", "train_resampling", "test_resampling", "NAs_in_predictions")) == 5,

                                     sum(unlist(lapply(perf[1:2], function(x) lapply(x, function(y) nrow(y) == TUNE_ITERS)))) == length(lst_obj) * 2, sum(unlist(lapply(perf[3:4], function(x) nrow(x) == length(lst_obj)))) == 2,

                                     length(perf$NAs_in_predictions) == length(lst_obj)))

                    testthat::expect_true(sum(res_out) == 6)
})


testthat::test_that("in CLASSIFICATION the performance_measures function returns : a list, the length of the list is 5, the names of the list are correct,

                    the rows of the 1st and 2nd sublists equals the number of tune_iters, the rows of the 3rd and 4th sublist equals the number of the algorithms used,

                    and the length of the last sublist equals the number of algorithms used", {

                    lst_obj = list(algs3 = algs3, algs4 = algs4, algs5 = algs5, algs6 = algs6)

                    perf = performance_measures(list_objects = lst_obj, eval_metric = acc, sort = list(variable = 'Median', decreasing = TRUE))

                    res_out = unlist(list(is.list(perf), length(perf) == 5, sum(names(perf) %in% c("train_params", "test_params", "train_resampling", "test_resampling", "NAs_in_predictions")) == 5,

                                          sum(unlist(lapply(perf[1:2], function(x) lapply(x, function(y) nrow(y) == TUNE_ITERS)))) == length(lst_obj) * 2, sum(unlist(lapply(perf[3:4], function(x) nrow(x) == length(lst_obj)))) == 2,

                                          length(perf$NAs_in_predictions) == length(lst_obj)))

                      testthat::expect_true(sum(res_out) == 6)
})


# use various summary functions to sort (besides the Median)

testthat::test_that("in REGRESSION the performance_measures function returns : a list, the length of the list is 5, the names of the list are correct,

                    the rows of the 1st and 2nd sublists equals the number of tune_iters, the rows of the 3rd and 4th sublist equals the number of the algorithms used,

                    and the length of the last sublist equals the number of algorithms used", {

                      lst_obj = list(algs = algs, algs1 = algs1, algs2 = algs2)

                      perf = performance_measures(list_objects = lst_obj, eval_metric = mse, sort = list(variable = 'Min.', decreasing = TRUE))

                      res_out = unlist(list(is.list(perf), length(perf) == 5, sum(names(perf) %in% c("train_params", "test_params", "train_resampling", "test_resampling", "NAs_in_predictions")) == 5,

                                            sum(unlist(lapply(perf[1:2], function(x) lapply(x, function(y) nrow(y) == TUNE_ITERS)))) == length(lst_obj) * 2, sum(unlist(lapply(perf[3:4], function(x) nrow(x) == length(lst_obj)))) == 2,

                                            length(perf$NAs_in_predictions) == length(lst_obj)))

                      testthat::expect_true(sum(res_out) == 6)
})


testthat::test_that("in CLASSIFICATION the performance_measures function returns : a list, the length of the list is 5, the names of the list are correct,

                    the rows of the 1st and 2nd sublists equals the number of tune_iters, the rows of the 3rd and 4th sublist equals the number of the algorithms used,

                    and the length of the last sublist equals the number of algorithms used", {

                      lst_obj = list(algs3 = algs3, algs4 = algs4, algs5 = algs5, algs6 = algs6)

                      perf = performance_measures(list_objects = lst_obj, eval_metric = acc, sort = list(variable = 'Max.', decreasing = TRUE))

                      res_out = unlist(list(is.list(perf), length(perf) == 5, sum(names(perf) %in% c("train_params", "test_params", "train_resampling", "test_resampling", "NAs_in_predictions")) == 5,

                                            sum(unlist(lapply(perf[1:2], function(x) lapply(x, function(y) nrow(y) == TUNE_ITERS)))) == length(lst_obj) * 2, sum(unlist(lapply(perf[3:4], function(x) nrow(x) == length(lst_obj)))) == 2,

                                            length(perf$NAs_in_predictions) == length(lst_obj)))

                      testthat::expect_true(sum(res_out) == 6)
})

