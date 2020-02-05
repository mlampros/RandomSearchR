
data(ionosphere, package = 'kknn')
ionosphere = ionosphere[, -2]           # remove second column which has a single unique value
data(Boston, package = 'MASS')
data(iris)

context('Utils testing')


# acc - function

testthat::test_that("the acc function returns a correct result", {

  y_true = c(1,1,1,0,0,0,2,2,2,3,3,3,3)
  preds = model.matrix(~. - 1, data.frame(as.factor(c(1,1,1,0,0,0,2,2,2,3,3,3,3))))

  res = acc(y_true, preds)

  testthat::expect_identical(res, 1)
})


# function_grid

testthat::test_that("the function_grid samples one item from each sublist, so that the length of the initial sublists is equal to the length of the end sublists", {

  lst_all = list(

    args1 = list(ntree = seq(30, 50, 5), mtry = c(2:3), nodesize = seq(5, 15, 5)),
    args2 = list(k = 3:20, distance = c(1:5), kernel = c("triweight", "cos", "inv", "gaussian", "rank", "optimal")),
    args3 = list(k = 3:20),
    args4 = list(weights_function = c('uniform', 'triangular', 'epanechnikov', 'biweight', 'triweight'), params_knn = list(k = 3:20, algorithm = c("kd_tree", "cover_tree"))),
    args5 = list(weights_function = c('uniform', 'triangular', 'epanechnikov', 'biweight'),
                 params_knn = list(k = 3:20, treetype = "kd", searchtype = c("standard", "priority"),
                                   radius = seq(3.0, 20.0, 0.5), eps = c(rep(0, 20), seq(0, 50, 1)), distance_metric = c('euclidean', 'manhattan'))),
    args6 = list(weights_function = c('uniform', 'triangular', 'epanechnikov', 'biweight'),
                 params_knn = list(k = 3:20, eps = c(rep(0, 20), seq(0, 50, 1)), searchtype = c("auto", "brute"))),
    args7 = list(control = RWeka::Weka_control(K = seq(3, 20, 1), I = c(TRUE, FALSE), X = c(TRUE, FALSE))),
    args8 = list(weights_function = c('uniform', 'triangular', 'epanechnikov', 'biweight', 'triweight'), k = 3:20, h = c(0.1, 0.5, 1.0, 2.0),
                 method = c('euclidean', 'manhattan')))

    res = as.vector(unlist(lapply(lst_all, function(x) length(function_grid(grid_params = x)))))
    length_sublists = as.vector(unlist(lapply(lst_all, function(x) length(x))))

  testthat::expect_identical(res, length_sublists)
})


testthat::test_that("the function_grid samples and returns a list for each grid_argument", {

  lst_all = list(
    args1 = list(ntree = seq(30, 50, 5), mtry = c(2:3), nodesize = seq(5, 15, 5)),
    args2 = list(k = 3:20, distance = c(1:5), kernel = c("triweight", "cos", "inv", "gaussian", "rank", "optimal")),
    args3 = list(k = 3:20),
    args4 = list(weights_function = c('uniform', 'triangular', 'epanechnikov', 'biweight', 'triweight'), params_knn = list(k = 3:20, algorithm = c("kd_tree", "cover_tree"))),
    args5 = list(weights_function = c('uniform', 'triangular', 'epanechnikov', 'biweight'),
                 params_knn = list(k = 3:20, treetype = "kd", searchtype = c("standard", "priority"),
                                   radius = seq(3.0, 20.0, 0.5), eps = c(rep(0, 20), seq(0, 50, 1)), distance_metric = c('euclidean', 'manhattan'))),
    args6 = list(weights_function = c('uniform', 'triangular', 'epanechnikov', 'biweight'),
                 params_knn = list(k = 3:20, eps = c(rep(0, 20), seq(0, 50, 1)), searchtype = c("auto", "brute"))),
    args7 = list(control = RWeka::Weka_control(K = seq(3, 20, 1), I = c(TRUE, FALSE), X = c(TRUE, FALSE))),
    args8 = list(weights_function = c('uniform', 'triangular', 'epanechnikov', 'biweight', 'triweight'), k = 3:20, h = c(0.1, 0.5, 1.0, 2.0),
                 method = c('euclidean', 'manhattan')))

  res = unlist(lapply(lst_all, function(x) is.list(function_grid(grid_params = x))))

  testthat::expect_true(sum(res) == length(lst_all))
})



# func_weka_rerun    [ for the RWeka package, if re-run = TRUE ]

testthat::test_that("the func_weka_rerun function takes a list of parameters (those parameters result from the subset_mods function) and it returns a list of sublists", {

  output_of_subset_mods_function = list(K = c(19, 13, 20, 6, 12),
                                          I = c("FALSE", "TRUE", "TRUE", "TRUE", "TRUE"),
                                          X = c("TRUE", "TRUE", "TRUE", "TRUE", "TRUE"))

  res = func_weka_rerun(output_of_subset_mods_function)

  len_each_sublist = sum(unlist(lapply(res, function(x) length(x[[1]]))))
  len_initial = sum(unlist(lapply(output_of_subset_mods_function, length)))
  len_list = length(res)
  len_init_each = mean(unlist(lapply(output_of_subset_mods_function, length)))


  testthat::expect_true(len_each_sublist == len_initial && len_list == len_init_each)
})


# optimal_grid  [ all functions except for RWeka, if re-run = TRUE ]

testthat::test_that("the func_weka_rerun function takes a list of parameters (those parameters result from the subset_mods function) and it returns a list of sublists", {

  output_of_subset_mods_function = list(k = c(3, 4, 4, 3, 7),
                                        distance = c(1, 1, 1, 1, 1),
                                        kernel = c("rank", "triangular","biweight", "cos", "cos"))

  res = lapply(1:length(output_of_subset_mods_function[[1]]), function(x) optimal_grid(output_of_subset_mods_function, x))

  len_each_sublist = sum(unlist(lapply(res, function(x) length(x))))
  len_initial = sum(unlist(lapply(output_of_subset_mods_function, length)))
  len_list = length(res)
  len_init_each = mean(unlist(lapply(output_of_subset_mods_function, length)))


  testthat::expect_true(len_each_sublist == len_initial && len_list == len_init_each)
})


# length_grid  [ independently of which algorithm will be used ]

testthat::test_that("the length_grid function returns the maximum length of the parameter grid", {

  output_of_subset_mods_function = list(k = c(3, 4, 4, 3, 7),
                                        distance = c(1, 1, 1, 1, 1),
                                        kernel = c("rank", "triangular","biweight", "cos", "cos"))

  res = max(length_grid(output_of_subset_mods_function))

  tes = max(unlist(lapply(output_of_subset_mods_function, length)))

  testthat::expect_true(res == tes)
})


# predict.gbm

testthat::test_that("the work-around for the gbm prediction function returns mutli-column probability matrix in case of multinomial classification", {

  fit = gbm::gbm(Species ~., iris, distribution = "multinomial", n.trees = 100)

  pred = predict_gbm_workaround_probs(fit, iris[, -ncol(iris)], type = 'response')

  testthat::expect_true(ncol(pred) == length(unique(iris[, 'Species'])))
})


testthat::test_that("the work-around for the gbm prediction function returns a two column probability matrix in case of binomial classification", {

  X = iris
  X$Species = as.numeric(X$Species)
  X$Species[X$Species == 1] = 2
  X$Species = X$Species - 2

  fit = gbm::gbm(Species ~., X, distribution = "bernoulli", n.trees = 100)

  pred = predict_gbm_workaround_probs(fit, X[, -ncol(X)], type = 'response')

  testthat::expect_true(ncol(pred) == length(unique(X[, 'Species'])))
})


# EXCEPTIONS_preds function

testthat::test_that("the output of the predictions produced by the selected REGRESSION algorithms is in the correct form", {

  X = Boston[, -ncol(Boston)]
  y = Boston[, ncol(Boston)]

  fit1 = nnet::nnet(medv~., Boston, size = 5, trace = F)
  pred1 = EXCEPTIONS_preds(fit1, Boston[, -ncol(Boston)], T)

#   fit2 = ranger::ranger(medv~., Boston, num.trees = 10, write.forest = T)
#   pred2 = preds = EXCEPTIONS_preds(fit2, Boston[, -ncol(Boston)], T)

  fit3 = e1071::svm(medv~., Boston)
  pred3 = EXCEPTIONS_preds(fit3, Boston[, -ncol(Boston)], T)

  fit4 = LiblineaR::LiblineaR(X, y, type = 11)
  pred4 = EXCEPTIONS_preds(fit4, Boston[, -ncol(Boston)], T)

  #lst = list(pred1, pred2, pred3, pred4)
  lst = list(pred1, pred3, pred4)

  res = unlist(lapply(lst, function(x) is.vector(x) && length(x) == nrow(Boston)))

  testthat::expect_true(sum(res) == length(lst))
})


testthat::test_that("the output of the predictions produced by the selected BINARY CLASSIFICATION algorithms is in the correct form", {

  X = ionosphere[, -ncol(ionosphere)]
  y = ionosphere[, ncol(ionosphere)]
  dat_gbm = ionosphere
  dat_gbm$class = as.numeric(dat_gbm$class) - 1

  fit1 = adabag::boosting(class~., ionosphere, mfinal = 5)
  pred1 = EXCEPTIONS_preds(fit1, ionosphere[, -ncol(ionosphere)], F)

  fit2 = MASS::lda(class~., ionosphere)
  pred2 = EXCEPTIONS_preds(fit2, ionosphere[, -ncol(ionosphere)], F)

  fit3 = e1071::naiveBayes(class~., ionosphere)
  pred3 = EXCEPTIONS_preds(fit3, ionosphere[, -ncol(ionosphere)], F)

  fit4 = extraTrees::extraTrees(X, y, ntree = 5)
  pred4 = EXCEPTIONS_preds(fit4, ionosphere[, -ncol(ionosphere)], F)

  fit5 = gbm::gbm(class~., data = dat_gbm, distribution = "bernoulli", n.trees = 5)                 # gbm-bernoulli requires the data to be in c(0,1)
  pred5 = EXCEPTIONS_preds(fit5, dat_gbm[, -ncol(dat_gbm)], F)

  fit6 = glmnet::cv.glmnet(as.matrix(X), y, type.measure = "class", family = 'binomial', nfolds = 3)       # with type = 'response' returns NA's in the glass-dataset
  pred6 = EXCEPTIONS_preds(fit6, as.matrix(ionosphere[, -ncol(ionosphere)]), F)

  fit7 = LiblineaR::LiblineaR(X, y, type = 0)                           # Computing probabilities is only supported for Logistic Regressions (LiblineaR 'type' 0, 6 or 7)
  pred7 = EXCEPTIONS_preds(fit7, ionosphere[, -ncol(ionosphere)], F)

  fit8 = e1071::svm(class~., ionosphere, probability = T)
  pred8 = EXCEPTIONS_preds(fit8, ionosphere[, -ncol(ionosphere)], F)

  fit9 = nnet::nnet(class~., ionosphere, size = 5, trace = F)
  pred9 = EXCEPTIONS_preds(fit9, ionosphere[, -ncol(ionosphere)], F)

#   fit10 = ranger::ranger(class~., ionosphere, num.trees = 10, write.forest = T, probability = T)
#   pred10 = preds = EXCEPTIONS_preds(fit10, ionosphere[, -ncol(ionosphere)], F)

  #lst = list(pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10)
  lst = list(pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9)

  res = unlist(lapply(lst, function(x) is.matrix(x) && ncol(x) == length(unique(ionosphere[, 'class']))))

  testthat::expect_true(sum(res) == length(lst))
})


testthat::test_that("the output of the predictions produced by the selected MULTICLASS CLASSIFICATION algorithms is in the correct form", {

  X = iris[, -ncol(iris)]
  y = iris[, ncol(iris)]

  fit1 = adabag::boosting(Species~., iris, mfinal = 5)
  pred1 = EXCEPTIONS_preds(fit1, iris[, -ncol(iris)], F)

  fit2 = MASS::lda(Species~., iris)
  pred2 = EXCEPTIONS_preds(fit2, iris[, -ncol(iris)], F)

  fit3 = e1071::naiveBayes(Species~., iris)
  pred3 = EXCEPTIONS_preds(fit3, iris[, -ncol(iris)], F)

  fit4 = extraTrees::extraTrees(X, y, ntree = 5)
  pred4 = EXCEPTIONS_preds(fit4, iris[, -ncol(iris)], F)

  fit5 = gbm::gbm(Species~., data = iris, distribution = "multinomial", n.trees = 5)
  pred5 = EXCEPTIONS_preds(fit5, iris[, -ncol(iris)], F)

  fit6 = glmnet::cv.glmnet(as.matrix(X), y, type.measure="class", family = 'multinomial', nfolds = 3)
  pred6 = EXCEPTIONS_preds(fit6, as.matrix(iris[, -ncol(iris)]), F)

  fit7 = LiblineaR::LiblineaR(X, y, type = 0)                           # Computing probabilities is only supported for Logistic Regressions (LiblineaR 'type' 0, 6 or 7)
  pred7 = EXCEPTIONS_preds(fit7, iris[, -ncol(iris)], F)

  fit8 = e1071::svm(Species~., iris, probability = T)
  pred8 = EXCEPTIONS_preds(fit8, iris[, -ncol(iris)], F)

  fit9 = nnet::nnet(Species~., iris, size = 5, trace = F)
  pred9 = EXCEPTIONS_preds(fit9, iris[, -ncol(iris)], F)

#   fit10 = ranger::ranger(Species~., iris, num.trees = 10, write.forest = T, probability = T)
#   pred10 = preds = EXCEPTIONS_preds(fit10, iris[, -ncol(iris)], F)

  #lst = list(pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10)
  lst = list(pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9)

  res = unlist(lapply(lst, function(x) is.matrix(x) && ncol(x) == length(unique(iris[, 'Species']))))

  testthat::expect_true(sum(res) == length(lst))
})




# repeated_resampling function


testthat::test_that("the repeated_resampling function gives an error if the method is invalid", {

  y = Boston$medv

  testthat::expect_error(repeated_resampling(y, NULL, REPEATS = 2, sample_rate = 0.75, FOLDS = NULL))
})


testthat::test_that("the repeated_resampling function gives an error if the method is invalid", {

  y = Boston$medv

  testthat::expect_error(repeated_resampling(y, 'invalid', REPEATS = 2, sample_rate = 0.75, FOLDS = NULL))
})


testthat::test_that("the repeated_resampling function gives an error if the method is invalid", {

  y = Boston$medv

  testthat::expect_error(repeated_resampling(y, 1, REPEATS = 2, sample_rate = 0.75, FOLDS = NULL))
})


testthat::test_that("the repeated_resampling function gives an error if the method is invalid", {

  y = Boston$medv

  testthat::expect_error(repeated_resampling(y, 'invalid', REPEATS = 2, sample_rate = 0.75, FOLDS = NULL))
})


testthat::test_that("the bootstrap method works for a continuous response variable", {

  y = Boston$medv

  samp_rate = 0.75
  repeats = 2

  train_length = round(length(y) * samp_rate)
  # test_length = length(y) - train_length                 # in bootstraps the test-set lengths are not equal

  res = repeated_resampling(y, 'bootstrap', REPEATS = repeats, sample_rate = samp_rate, FOLDS = NULL)

  testthat::expect_true(is.list(res) && length(res) == repeats && mean(unlist(lapply(1:length(res), function(x) length(res$idx_train[[x]])))) == train_length)
})


testthat::test_that("the repeated_resampling function works for a factor response variable", {

  y = iris$Species

  samp_rate = 0.75
  repeats = 2

  train_length = round(length(y) * samp_rate)
  # test_length = length(y) - train_length                 # in bootstraps the test-set lengths are not equal

  res = repeated_resampling(y, 'bootstrap', REPEATS = repeats, sample_rate = samp_rate, FOLDS = NULL)

  testthat::expect_true(is.list(res) && length(res) == repeats && mean(unlist(lapply(1:length(res), function(x) length(res$idx_train[[x]])))) == train_length)
})


testthat::test_that("the train-test-split method works for a continuous response variable", {

  y = Boston$medv

  samp_rate = 0.75
  repeats = 2

  train_length = round(length(y) * samp_rate)
  test_length = length(y) - train_length

  res = repeated_resampling(y, 'train_test_split', REPEATS = repeats, sample_rate = samp_rate, FOLDS = NULL)

  testthat::expect_true(is.list(res) && length(res) == repeats && mean(unlist(lapply(1:length(res), function(x) length(res$idx_train[[x]])))) == train_length

                        && mean(unlist(lapply(1:length(res), function(x) length(res$idx_test[[x]])))) == test_length)
})


testthat::test_that("the train-test-split method works for a factor response variable", {

  y = iris$Species

  samp_rate = 0.75
  repeats = 2

  train_length = round(length(y) * samp_rate)
  test_length = length(y) - train_length

  res = repeated_resampling(y, 'train_test_split', REPEATS = repeats, sample_rate = samp_rate, FOLDS = NULL)

  testthat::expect_true(is.list(res) && length(res) == repeats && mean(unlist(lapply(1:length(res), function(x) length(res$idx_train[[x]])))) == train_length

                        && mean(unlist(lapply(1:length(res), function(x) length(res$idx_test[[x]])))) == test_length)
})


testthat::test_that("the cross_validation method works for a continuous response variable", {

  y = Boston$medv

  folds = 3
  repeats = 2

  res = repeated_resampling(y, 'cross_validation', REPEATS = repeats, sample_rate = NULL, FOLDS = folds)

  tmp_train = unlist(res$idx_train, recursive = F)
  
  tmp_test = unlist(res$idx_test, recursive = F)
  
  tmp = unlist(lapply(1:length(tmp_train), function(x) (length(tmp_train[[x]]) + length(tmp_test[[x]]))))

  testthat::expect_true(is.list(res) && length(res) == repeats && sum(sapply(tmp, function(x) identical(length(y), x))) == length(tmp))
})


testthat::test_that("the cross_validation method works for a factor response variable", {

  y = iris$Species

  folds = 3
  repeats = 2

  res = repeated_resampling(y, 'cross_validation', REPEATS = repeats, sample_rate = NULL, FOLDS = folds)

  tmp_train = unlist(res$idx_train, recursive = F)
  
  tmp_test = unlist(res$idx_test, recursive = F)
  
  tmp = unlist(lapply(1:length(tmp_train), function(x) (length(tmp_train[[x]]) + length(tmp_test[[x]]))))
  

  testthat::expect_true(is.list(res) && length(res) == repeats && sum(sapply(tmp, function(x) identical(length(y), x))) == length(tmp))
})


# shuffle data

testthat::test_that("shuffle data takes a vector as input and returns a vector as output", {

  y = c(1:50)

  testthat::expect_true(is.vector(FeatureSelection::func_shuffle(y, times = 10)))
})

testthat::test_that("the length of the input vector equals the length of the output vector", {

  y = c(1:50)

  output = FeatureSelection::func_shuffle(y, times = 10)

  testthat::expect_true(length(y) == length(output))
})


# classification folds

testthat::test_that("throws an error if the RESP is not a factor", {

  y = c(1:10)

  testthat::expect_error(FeatureSelection::class_folds(5, y, shuffle = T), "RESP must be a factor")
})


testthat::test_that("returns a warning if the folds are not equally split", {

  y = as.factor(sample(1:5, 99, replace = T))

  testthat::expect_warning(FeatureSelection::class_folds(5, y, shuffle = T), 'the folds are not equally split')
})

testthat::test_that("the number of folds equals the number of the resulted sublist indices", {

  y = as.factor(sample(1:5, 100, replace = T))

  testthat::expect_length(FeatureSelection::class_folds(5, y, shuffle = T), 5)
})


# return object with shuffle = F

testthat::test_that("the number of folds equals the number of the resulted sublist indices", {

  y = as.factor(sample(1:5, 100, replace = T))

  testthat::expect_length(FeatureSelection::class_folds(5, y, shuffle = F), 5)
})


# regression folds

testthat::test_that("throws an error if the RESP is not a factor", {

  y = as.factor(c(1:50))

  testthat::expect_error(FeatureSelection::regr_folds(5, y, stratified = F), "this function is meant for regression for classification use the 'class_folds' function")
})


testthat::test_that("returns a warning if the folds are not equally split", {

  y = sample(1:5, 99, replace = T)

  testthat::expect_warning(FeatureSelection::regr_folds(5, y, stratified = F), 'the folds are not equally split')
})

testthat::test_that("the number of folds equals the number of the resulted sublist indices", {

  y = sample(1:5, 100, replace = T)

  testthat::expect_length(FeatureSelection::regr_folds(5, y, stratified = F), 5)
})

# object with stratified = T

testthat::test_that("the number of folds equals the number of the resulted sublist indices", {

  y = sample(1:5, 100, replace = T)

  testthat::expect_length(FeatureSelection::regr_folds(5, y, stratified = T), 5)
})
