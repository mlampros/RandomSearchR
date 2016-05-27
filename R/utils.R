
#' shuffle data
#'
#' this function shuffles the items of a vector
#' @keywords internal

func_shuffle = function(vec, times = 10) {

  for (i in 1:times) {

    out = sample(vec, length(vec))
  }
  out
}


#' stratified folds (in classification)                      [ detailed information about class_folds in the FeatureSelectionR package ]
#'
#' this function creates stratified folds in binary and multiclass classification
#' @keywords internal
#' @importFrom utils combn


class_folds = function(folds, RESP, shuffle = FALSE) {

  if (!is.factor(RESP)) {

    stop(simpleError("RESP must be a factor"))
  }

  clas = lapply(unique(RESP), function(x) which(RESP == x))

  len = lapply(clas, function(x) length(x))

  samp_vec = rep(1/folds, folds)

  prop = lapply(len, function(y) sapply(1:length(samp_vec), function(x) round(y * samp_vec[x])))

  repl = unlist(lapply(prop, function(x) sapply(1:length(x), function(y) rep(paste0('fold_', y), x[y]))))

  spl = suppressWarnings(split(1:length(RESP), repl))

  sort_names = paste0('fold_', 1:folds)

  spl = spl[sort_names]

  if (length(table(unlist(lapply(spl, function(x) length(x))))) > 1) {

    warning('the folds are not equally split')            # the warning appears when I divide the unique labels to the number of folds and instead of an integer I get a float
  }

  if (shuffle == TRUE) {

    spl = lapply(spl, function(x) func_shuffle(x))           # the indices of the unique levels will be shuffled
  }

  ind = t(combn(1:folds, 2))

  ind1 = apply(ind, 1, function(x) length(intersect(spl[x[1]], spl[x[2]])))

  if (sum(ind1) > 0) {

    stop(simpleError("there is an intersection between the resulted indexes of the folds"))

  }

  if (length(unlist(spl)) != length(RESP)) {

    stop(simpleError("the number of items in the folds are not equal with the response items"))
  }

  spl
}


#' create folds (in regression)                                           [ detailed information about class_folds in the FeatureSelectionR package ]
#'
#' this function creates both stratified and non-stratified folds in regression
#' @keywords internal


regr_folds = function(folds, RESP, stratified = FALSE) {

  if (is.factor(RESP)) {

    stop(simpleError("this function is meant for regression for classification use the 'class_folds' function"))
  }

  samp_vec = rep(1/folds, folds)

  sort_names = paste0('fold_', 1:folds)


  if (stratified == TRUE) {

    stratif = cut(RESP, breaks = folds)

    clas = lapply(unique(stratif), function(x) which(stratif == x))

    len = lapply(clas, function(x) length(x))

    prop = lapply(len, function(y) sapply(1:length(samp_vec), function(x) round(y * samp_vec[x])))

    repl = unlist(lapply(prop, function(x) sapply(1:length(x), function(y) rep(paste0('fold_', y), x[y]))))

    spl = suppressWarnings(split(1:length(RESP), repl))}

  else {

    prop = lapply(length(RESP), function(y) sapply(1:length(samp_vec), function(x) round(y * samp_vec[x])))

    repl = func_shuffle(unlist(lapply(prop, function(x) sapply(1:length(x), function(y) rep(paste0('fold_', y), x[y])))))

    spl = suppressWarnings(split(1:length(RESP), repl))
  }

  spl = spl[sort_names]

  if (length(table(unlist(lapply(spl, function(x) length(x))))) > 1) {

    warning('the folds are not equally split')            # the warning appears when I divide the unique labels to the number of folds and instead of an ingeger I get a float
  }

  if (length(unlist(spl)) != length(RESP)) {

    stop(simpleError("the length of the splits are not equal with the length of the response"))
  }

  spl
}


#' evaluation metric for regression
#' @keywords internal
#' @export

mse = function(y_true, y_pred) {

  out = mean((y_true - y_pred) ^ 2)

  out
}


#' evaluation metric for binary and multiclass classification
#' @keywords internal
#' @export

acc = function(y_true, preds) {

  out = table(y_true, max.col(preds, ties.method = "random"))

  acc = sum(diag(out))/sum(out)

  acc
}


#' grid function  [ this function takes a grid of parameters, IF the grid includes a list and this list includes vectors then it takes a sample of 1 else if it includes a single value then it returns this value ]
#' @keywords internal

function_grid = function(grid_params = NULL) {

  if ('control' %in% names(grid_params) && class(grid_params$control) == "Weka_control") {         # exception on sampling parameters for RWeka

    lst = list()

    for (i in 1:length(names(grid_params$control))) {

      lst[[i]] = paste(names(grid_params$control)[i], if (length(grid_params$control[[names(grid_params$control)[i]]]) > 1) sample(grid_params$control[[names(grid_params$control)[i]]], 1) else grid_params$control[[names(grid_params$control)[i]]], sep = ' = ')
    }

    exp = paste(unlist(lst), collapse = ', ')

    Args = list(control = eval(parse(text = paste0('RWeka::Weka_control(', paste0(exp, ')')))))}

  else {

    Args = lapply(grid_params, function(x) if (is.list(x)) lapply(x, function(y) if (length(y) > 1) sample(y, size = 1) else y) else if (length(x) > 1) sample(x, size = 1) else x)
  }

  Args
}


#' secondary function for weka-algos (if re-run = TRUE)
#' @keywords internal
#' @importFrom RWeka Weka_control

func_weka_rerun = function(bst_lst) {

  nams = names(bst_lst)

  pas = list()

  for (i in 1:length(nams)) {

    pas[[nams[i]]] = paste(nams[i], bst_lst[[i]], sep = ' = ')
  }

  tmp = do.call(cbind, pas)

  tmp1 = lapply(1:dim(tmp)[1], function(x) paste(as.vector(tmp[x, ]), collapse = ', '))


  out = lapply(tmp1, function(x) list(control = eval(parse(text = paste0('RWeka::Weka_control(', paste0(x, ')'))))))

  out
}




#' function that will be used if re-run = TRUE
#' @keywords internal

optimal_grid = function(grid_params = NULL, iter = NULL) {

  Args = lapply(grid_params, function(x) if (is.list(x)) lapply(x, function(y) if (length(y) > 1) y[iter] else y) else if (length(x) > 1) x[iter] else x)

  Args
}


#' function to get the max. length of the parameter-grid
#' @keywords internal

length_grid = function(grid_params = NULL) {

  Args = lapply(grid_params, function(x) if (is.list(x)) lapply(x, function(y) length(y)) else length(x))

  as.vector(unlist(Args))
}


#' [  http://www.samuelbosch.com/2015/09/workaround-ntrees-is-missing-in-r.html,  with minor modifications ]
#' work around for bug in gbm 2.1.1 -- PREDICTION-PROBABILITIES when predict(fit, data, type = 'response')
#' @keywords internal
#' @importFrom gbm gbm.perf
#' @importFrom gbm predict.gbm

predict.gbm <- function (object, newdata, n.trees, type = "link", single.tree = FALSE, ...) {

  if (missing(n.trees)) {

    if (object$train.fraction < 1) {

      n.trees <- gbm::gbm.perf(object, method = "test", plot.it = FALSE)
    }

    else if (!is.null(object$cv.error)) {

      n.trees <- gbm::gbm.perf(object, method = "cv", plot.it = FALSE)
    }

    else {

      n.trees <- length(object$train.error)
    }

    #cat(paste("Using", n.trees, "trees...\n"))
    gbm::predict.gbm(object, newdata, n.trees, type, single.tree, ...)
  }
}


#' secondary function [ see previous one ]
#' @keywords internal

predict_gbm_workaround_probs <- function(object, newdata, n.trees, type = "link", single.tree = FALSE, ...) {        # don't change anything, otherwise rstudio crashes

  if (object$distribution$name == 'multinomial' && type == 'response') {                                 # multinomial probs

    out = predict.gbm(object, newdata, n.trees, type = type, single.tree = single.tree, ...)
    out = matrix(out, ncol = length(out)/dim(newdata)[1], nrow = dim(newdata)[1], byrow = FALSE)
    }

  else if (object$distribution$name != 'multinomial' && type == 'response') {                             # binomial probs

    tmp_out = predict.gbm(object, newdata, n.trees, type = type, single.tree = single.tree, ...)
    tmp = matrix(tmp_out, ncol = length(tmp_out)/dim(newdata)[1], nrow = dim(newdata)[1], byrow = FALSE)
    out = cbind(matrix(1.0 - tmp, ncol = 1), tmp)}

  else {

    out = predict.gbm(object, newdata, n.trees, type = type, single.tree = single.tree, ...)
  }

  return(out)
}



#' EXCEPTIONS in predictions [ REGRESSION and CLASSIFICATION ]
#' @keywords internal


EXCEPTIONS_preds = function(FIT, DATA, regression) {

  if (regression == FALSE) {

    potential_class = c("boosting", "lda", "naiveBayes", "extraTrees", "gbm", "cv.glmnet", "LiblineaR", "svm", "nnet", "ranger")}

  else {

    potential_class = c("svm", "nnet", "ranger", "LiblineaR")
  }

  tmp_class = class(FIT)

  if (length(tmp_class) > 1) tmp_class = tmp_class[2]                                                   # exception for a single algorithm that returns more than one class

  if (tmp_class %in% potential_class) {

    idx_nams = which(potential_class == tmp_class)

    if (regression == TRUE) {

      switch(potential_class[idx_nams],

             nnet = {preds = predict(FIT, DATA)[, 1]},

             ranger = {preds = predict(FIT, DATA)$predictions},

             svm = {preds = predict(FIT, DATA)},

             LiblineaR = {preds = predict(FIT, DATA)$predictions}
      )}

    else {

      switch(potential_class[idx_nams],

             lda = {preds = predict(FIT, DATA, type = 'prob')$posterior},

             extraTrees = {preds = predict(FIT, DATA, probability = TRUE)},

             naiveBayes = {preds = predict(FIT, DATA, type = 'raw')},

             nnet = {preds = predict(FIT, DATA, type = 'raw');

                     if (ncol(preds) == 1) { preds = cbind(matrix(1 - preds, ncol = 1), preds) }},

             boosting = {preds = predict(FIT, DATA, type = 'prob')$prob},

             ranger = {preds = predict(FIT, DATA)$predictions},

             gbm = {preds = predict_gbm_workaround_probs(FIT, DATA, type = 'response')},

             cv.glmnet = {preds = predict(FIT, DATA, s = FIT$lambda.min, type = 'response');

                          preds = matrix(preds, ncol = round(length(preds)/dim(DATA)[1]), nrow = dim(DATA)[1]);

                          if (ncol(preds) == 1) { preds = cbind(matrix(1 - preds, ncol = 1), preds) } },               # exception for binary classification

             LiblineaR = {preds = predict(FIT, DATA, proba = TRUE)$probabilities},                                    # Computing probabilities is only supported for Logistic Regressions (LiblineaR 'type' 0, 6 or 7)

             svm = {preds = attr(predict(FIT, DATA, probability = TRUE), "prob")}
      )}
  }

  else {

    if (regression == FALSE) preds = predict(FIT, DATA, type = 'prob') else preds = predict(FIT, DATA)
  }

  preds
}



#' Resampling methods
#'
#' Three different methods : 'bootstrap', 'train_test_split' and 'cross_validation'
#'
#' @param y the response variable (continuous or factor)
#' @param method one of 'bootstrap', 'train_test_split', 'cross_validation'
#' @param iter defaults to 1 and can vary when the resampling_methods function is used internally (as is the case in the random_search_resample function)
#' @param REPEATS the number of times that the method should be repeated
#' @param sample_rate train-data-sample-rate for 'bootstrap' and 'train_test_split'
#' @param FOLDS the number of folds for the 'cross_validation' method
#' @details
#' This function is meant for splitting the data using three resampling methods, with the option of multiple repeats.
#' @export
#' @examples
#'
#' # data(Boston, package = 'MASS')
#' # y = Boston$medv
#' # res = resampling_methods(y, 'bootstrap', 3, REPEATS = 2, sample_rate = 0.75, FOLDS = NULL)
#'


resampling_methods = function(y, method, iter = 1, REPEATS = 1, sample_rate = NULL, FOLDS = NULL) {

  if (is.null(iter)) stop('iter should be any positive or negative value except for NULL')
  if (!method %in% c('bootstrap', 'train_test_split', 'cross_validation')) stop('invalid method. Choose one of bootstrap, train_test_split, cross_validation')

  method = match.arg(method, c('bootstrap', 'train_test_split', 'cross_validation'))

  switch(method,

         bootstrap = {

           idx_train_lst = idx_test_lst = list()

           for (Repeat in 1:REPEATS) {

             set.seed(iter * Repeat)
             tmp_train_idx = sample(1:length(y), size = round(length(y) * sample_rate), replace = TRUE)      # in case of bootstrap the test splits do not have the same number of observations as the sampling is with replacement
             idx_train_lst[[Repeat]] = tmp_train_idx
             idx_test_lst[[Repeat]] = setdiff(1:length(y), tmp_train_idx) }
         },

         train_test_split = {

           idx_train_lst = idx_test_lst = list()

           for (Repeat in 1:REPEATS) {

             set.seed(iter * Repeat)
             tmp_train_idx = sample(1:length(y), size = round(length(y) * sample_rate), replace = FALSE)
             idx_train_lst[[Repeat]] = tmp_train_idx
             idx_test_lst[[Repeat]] = setdiff(1:length(y), tmp_train_idx) }
         },

         cross_validation = {

           if (is.factor(y)) {

             set.seed(iter)
             idx_test_lst = class_folds(FOLDS, y)
             idx_train_lst = lapply(1:length(idx_test_lst), function(x) unlist(idx_test_lst[-x]))}

           else {

             set.seed(iter)
             idx_test_lst = regr_folds(FOLDS, y)
             idx_train_lst = lapply(1:length(idx_test_lst), function(x) unlist(idx_test_lst[-x]))}
         }
  )

  return(list(idx_train = idx_train_lst, idx_test = idx_test_lst))
}


