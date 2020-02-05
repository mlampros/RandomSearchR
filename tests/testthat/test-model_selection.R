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
ionosphere$V1 = as.numeric(ionosphere$V1)
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


#=========================================================================================================================================================

# performance measures objects for regression and classification

lst_obj_REGRESSION = list(algs = algs, algs1 = algs1, algs2 = algs2)

perf_REGRESSION = performance_measures(list_objects = lst_obj_REGRESSION, eval_metric = mse, sort = list(variable = 'Median', decreasing = TRUE))


lst_obj_CLASSIFICATION = list(algs3 = algs3, algs4 = algs4, algs5 = algs5, algs6 = algs6)

perf_CLASSIFICATION = performance_measures(list_objects = lst_obj_CLASSIFICATION, eval_metric = acc, sort = list(variable = 'Median', decreasing = TRUE))

#==========================================================================================================================================================

# subset models

BST_MODS = 2

res_REGRESSION = subset_mods(perf_meas_OBJ = perf_REGRESSION, bst_mods = BST_MODS, train_params = FALSE)

res_CLASSIFICATION = subset_mods(perf_meas_OBJ = perf_CLASSIFICATION, bst_mods = BST_MODS, train_params = TRUE)

#==========================================================================================================================================================

# Re-run algorithms with the best resulted parameters

# REGRESSION

algs = random_search_resample(y1, tune_iters = TUNE_ITERS,                                                                               # makes no difference here As the parameter 're_run_params' = TRUE specifies the number of iters

                              resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                              ALGORITHM = list(package = require(kknn), algorithm = kknn),

                              grid_params = res_REGRESSION$algs,                                       # UPDATE THE 'grid_params' WITH THE bst_mods SELECTED MODELS

                              DATA = list(formula = form, train = ALL_DATA),

                              Args = NULL,

                              regression = TRUE, re_run_params = T)


algs1 = random_search_resample(y1, tune_iters = TUNE_ITERS,                                                                              # makes no difference here As the parameter 're_run_params' = TRUE specifies the number of iters

                               resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                               ALGORITHM = list(package = require(caret), algorithm = knnreg),

                               grid_params = res_REGRESSION$algs1,                                    # UPDATE THE 'grid_params' WITH THE bst_mods SELECTED MODELS

                               DATA = list(x = X, y = y1),

                               Args = NULL,

                               regression = TRUE, re_run_params = T)


algs2 = random_search_resample(y1, tune_iters = TUNE_ITERS,                                                                              # makes no difference here As the parameter 're_run_params' = TRUE specifies the number of iters

                               resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 3),

                               ALGORITHM = list(package = require(RWeka), algorithm = IBk),

                               grid_params = res_REGRESSION$algs2,                                   # UPDATE THE 'grid_params' WITH THE bst_mods SELECTED MODELS

                               DATA = list(formula = form, data = ALL_DATA),

                               Args = NULL,

                               regression = TRUE, re_run_params = T)



# CLASSIFICATION

algs3 = random_search_resample(as.factor(y1_class), tune_iters = TUNE_ITERS,                                  # makes no difference here As the parameter 're_run_params' = TRUE specifies the number of iters

                               resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                               ALGORITHM = list(package = require(RWeka), algorithm = AdaBoostM1),

                               grid_params = res_CLASSIFICATION$algs3,                                       # UPDATE THE 'grid_params' WITH THE bst_mods SELECTED MODELS

                               DATA = list(formula = form_class, data = ALL_DATA_class),

                               Args = NULL,

                               regression = F, re_run_params = T)


algs4 = random_search_resample(as.factor(y1_class), tune_iters = TUNE_ITERS,                                  # makes no difference here As the parameter 're_run_params' = TRUE specifies the number of iters

                               resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                               ALGORITHM = list(package = require(gbm), algorithm = gbm),

                               grid_params = res_CLASSIFICATION$algs4,                               # UPDATE THE 'grid_params' WITH THE bst_mods SELECTED MODELS

                               DATA = list(formula = form_class, data = all_data_gbm),         # exception, as it requires the response to be in c(0,1)

                               Args = list(distribution = 'bernoulli'),

                               regression = F, re_run_params = T)


algs5 = random_search_resample(as.factor(y1_class), tune_iters = TUNE_ITERS,                                  # makes no difference here As the parameter 're_run_params' = TRUE specifies the number of iters

                               resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                               ALGORITHM = list(package = require(h2o), algorithm = h2o.deeplearning),

                               grid_params = res_CLASSIFICATION$algs5,                                 # UPDATE THE 'grid_params' WITH THE bst_mods SELECTED MODELS

                               DATA = list(h2o.x = X_class, h2o.y = as.factor(y1_class)),          # response should be a factor

                               Args = list(distribution = 'bernoulli', hidden = c(10, 10)),

                               regression = F, re_run_params = T)


algs6 = random_search_resample(as.factor(y1_class), tune_iters = TUNE_ITERS,                                 # makes no difference here As the parameter 're_run_params' = TRUE specifies the number of iters

                               resampling_method = list(method = 'train_test_split', repeats = 5, sample_rate = 2/3, folds = NULL),

                               ALGORITHM = list(package = library(xgboost), algorithm = xgb.train),

                               grid_params = res_CLASSIFICATION$algs6,                                          # UPDATE THE 'grid_params' WITH THE bst_mods SELECTED MODELS

                               DATA = list(watchlist = list(label = y1_class, data = X_class)),

                               Args = list(nrounds = 5, verbose = 0, print.every.n = 10, early.stop.round = 5, maximize = FALSE),

                               regression = F, re_run_params = T)



# resulted list of algorithms

new_lst_REGRESSION = list(algs = algs, algs1 = algs1, algs2 = algs2)

new_lst_CLASSIFICATION = list(algs3 = algs3, algs4 = algs4, algs5 = algs5, algs6 = algs6)

#==========================================================================================================================================================


context("Model selection")


# Error handling

testthat::test_that("if the evaluation metric is not in form of a string it returns an error", {

  testthat::expect_error(model_selection(new_lst_REGRESSION, on_Train = FALSE, regression = T, evaluation_metric = mse, t.test.conf.int = 0.95, cor_test = list(method = 'spearman'), sort_decreasing = TRUE))
})


testthat::test_that("if the evaluation metric is not in form of a string or NULL it returns an error", {

  testthat::expect_error(model_selection(new_lst_REGRESSION, on_Train = FALSE, regression = T, evaluation_metric = NULL, t.test.conf.int = 0.95, cor_test = list(method = 'spearman'), sort_decreasing = TRUE))
})


testthat::test_that("if the number of algorithms is less than 2 it returns an error", {

  testthat::expect_error(model_selection(list(algs = algs), on_Train = FALSE, regression = T, evaluation_metric = 'mse', t.test.conf.int = 0.95, cor_test = list(method = 'spearman'), sort_decreasing = TRUE))
})


# REGRESSION       [ here nrow(t(combn(1:length(new_lst_REGRESSION), 2))) are the possible combinations of the algorithms in the list ]

testthat::test_that("if the correlation test is NULL, if on_Train = F, if regression = T, it returns a NON-empty data.frame with 10 columns", {

  modse = model_selection(new_lst_REGRESSION, on_Train = FALSE, regression = T, evaluation_metric = 'mse', t.test.conf.int = 0.95, cor_test = NULL, sort_decreasing = TRUE)

  testthat::expect_true(is.data.frame(modse) && ncol(modse) == 10 && sum(dim(modse)) == nrow(t(combn(1:length(new_lst_REGRESSION), 2))) + 10)
})


testthat::test_that("if the correlation test is NOT NULL, if on_Train = F, if regression = T, it returns a NON-empty data.frame with 10 columns", {

  modse = model_selection(new_lst_REGRESSION, on_Train = FALSE, regression = T, evaluation_metric = 'mse', t.test.conf.int = 0.95, cor_test = list(method = 'spearman'), sort_decreasing = TRUE)

  testthat::expect_true(is.data.frame(modse) && ncol(modse) == 13 && sum(dim(modse)) == nrow(t(combn(1:length(new_lst_REGRESSION), 2))) + 13)
})


testthat::test_that("if the correlation test is NULL, if on_Train = T, if regression = T, it returns a NON-empty data.frame with 10 columns", {

  modse = model_selection(new_lst_REGRESSION, on_Train = T, regression = T, evaluation_metric = 'mse', t.test.conf.int = 0.95, cor_test = NULL, sort_decreasing = TRUE)

  testthat::expect_true(is.data.frame(modse) && ncol(modse) == 10 && sum(dim(modse)) == nrow(t(combn(1:length(new_lst_REGRESSION), 2))) + 10)
})


testthat::test_that("if the correlation test is NOT NULL, if on_Train = T, if regression = T, it returns a NON-empty data.frame with 13 columns", {

  modse = model_selection(new_lst_REGRESSION, on_Train = T, regression = T, evaluation_metric = 'mse', t.test.conf.int = 0.95, cor_test = list(method = 'spearman'), sort_decreasing = TRUE)

  testthat::expect_true(is.data.frame(modse) && ncol(modse) == 13 && sum(dim(modse)) == nrow(t(combn(1:length(new_lst_REGRESSION), 2))) + 13)
})


# CLASSIFICATION       [ here nrow(t(combn(1:length(new_lst_CLASSIFICATION), 2))) are the possible combinations of the algorithms in the list ]

testthat::test_that("if the correlation test is NULL, if on_Train = F, if regression = F, it returns a NON-empty data.frame with 10 columns", {

  modse = model_selection(new_lst_CLASSIFICATION, on_Train = FALSE, regression = F, evaluation_metric = 'acc', t.test.conf.int = 0.95, cor_test = NULL, sort_decreasing = TRUE)

  testthat::expect_true(is.data.frame(modse) && ncol(modse) == 10 && sum(dim(modse)) == nrow(t(combn(1:length(new_lst_CLASSIFICATION), 2))) + 10)
})



testthat::test_that("if the correlation test is NOT NULL, if on_Train = F, if regression = F, it returns a NON-empty data.frame with 13 columns", {

  modse = model_selection(new_lst_CLASSIFICATION, on_Train = FALSE, regression = F, evaluation_metric = 'acc', t.test.conf.int = 0.95, cor_test = list(method = 'spearman'), sort_decreasing = TRUE)

  testthat::expect_true(is.data.frame(modse) && ncol(modse) == 13 && sum(dim(modse)) == nrow(t(combn(1:length(new_lst_CLASSIFICATION), 2))) + 13)
})


testthat::test_that("if the correlation test is NULL, if on_Train = T, if regression = F, it returns a NON-empty data.frame with 10 columns", {

  modse = model_selection(new_lst_CLASSIFICATION, on_Train = T, regression = F, evaluation_metric = 'acc', t.test.conf.int = 0.95, cor_test = NULL, sort_decreasing = TRUE)

  testthat::expect_true(is.data.frame(modse) && ncol(modse) == 10 && sum(dim(modse)) == nrow(t(combn(1:length(new_lst_CLASSIFICATION), 2))) + 10)
})


testthat::test_that("if the correlation test is NOT NULL, if on_Train = T, if regression = F, it returns a NON-empty data.frame with 13 columns", {

  modse = model_selection(new_lst_CLASSIFICATION, on_Train = T, regression = F, evaluation_metric = 'acc', t.test.conf.int = 0.95, cor_test = list(method = 'spearman'), sort_decreasing = TRUE)

  testthat::expect_true(is.data.frame(modse) && ncol(modse) == 13 && sum(dim(modse)) == nrow(t(combn(1:length(new_lst_CLASSIFICATION), 2))) + 13)
})
