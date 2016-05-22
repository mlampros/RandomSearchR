#' model_selection
#'
#' compare pairs of models using statistics such as t.test, correlation, evaluation.

#' @param list_of_algos a list of model objects (first use the subset_mods to select the best and then re-run)
#' @param on_Train if TRUE, then it applies the test-statistics on train-data
#' @param regression is it a regression or a classification task
#' @param evaluation_metric one of the evaluation metrics (accuracy, rmse, etc.)
#' @param t.test.conf.int specify confidence interval for the t.test statistic (0.95, 0.99, etc.)
#' @param cor_test one of spearman, pearson, kendal
#' @param sort_decreasing sorts the resulted data.frame by the evaluation metric of the first algorithm in either increasing or decreasing order
#' @return a data frame
#' @details
#' This function takes a list of objects after they were subset and re-run on the same resampling method. It returns a data frame with statistics for each pair of them.
#' @export
#' @importFrom stats cor.test t.test
#' @examples
#'
#' # random-forest
#'
#'# res_rf = random_search_resample(as.factor(y1), tune_iters = 30,
#'#
#'#                                 resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
#'#
#'#                                 ALGORITHM = list(package = require(randomForest), algorithm = randomForest),
#'#
#'#                                 grid_params = bst_m$rf,
#'#
#'#                                 DATA = list(x = X, y = as.factor(y1)),
#'#
#'#                                 Args = NULL,
#'#
#'#                                 regression = FALSE, re_run_params = TRUE)
#'#
#'#
#'# RWeka Bagging
#'#
#'#
#'# res_logitBoost = random_search_resample(as.factor(y1), tune_iters = 30,
#'#
#'#                                         resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
#'#
#'#                                         ALGORITHM = list(package = require(RWeka), algorithm = LogitBoost),
#'#
#'#                                         grid_params = bst_m$logitboost_weka,
#'#
#'#                                         DATA = list(formula = form, data = ALL_DATA),
#'#
#'#                                         Args = NULL,
#'#
#'#                                         regression = FALSE, re_run_params = TRUE)
#'#
#'#
#'# tmp_lst = list(rf = res_rf, LogBoost = res_logitBoost)
#'#
#' #res = model_selection(tmp_lst, on_Train = FALSE, regression = FALSE, evaluation_metric = 'acc', t.test.conf.int = 0.95, cor_test = list(method = 'spearman'), sort_decreasing = TRUE)
#'
#'# res



model_selection = function(list_of_algos, on_Train = FALSE, regression = TRUE, evaluation_metric = NULL, t.test.conf.int = 0.95, cor_test = NULL, sort_decreasing = TRUE) {

  if (is.null(evaluation_metric) || !is.character(evaluation_metric)) stop(simpleError('the evaluation metric should be in form of a string and not NULL'))

  if (length(list_of_algos) < 2) stop(simpleError('the number of algorithms should be at least 2'))

  tmp_names = names(list_of_algos)

  if (on_Train) {           # use train-predictions for the t.test-statistic

    dat = lapply(list_of_algos, function(x) lapply(x$PREDS, function(y) lapply(y, function(e) e$pred_tr)))}

  else {

    dat = lapply(list_of_algos, function(x) lapply(x$PREDS, function(y) lapply(y, function(e) e$pred_te)))
  }

  if (on_Train) {

    if (!is.null(cor_test)) {

      if (regression) {

        dat_cor = lapply(list_of_algos, function(x) lapply(x$PREDS, function(y) lapply(y, function(e) e$pred_tr)))}

      else {

        dat_cor = lapply(list_of_algos, function(x) unlist(lapply(x$PREDS, function(y) lapply(y, function(e) max.col(e$pred_tr))), recursive = FALSE))
      }
    }

    dat = lapply(list_of_algos, function(x) unlist(lapply(x$PREDS, function(y) lapply(y, function(e) do.call(evaluation_metric, list(e$y_tr, e$pred_tr))))))}

  else {

    if (!is.null(cor_test)) {

      if (regression) {

        dat_cor = lapply(list_of_algos, function(x) lapply(x$PREDS, function(y) lapply(y, function(e) e$pred_te)))}

      else {

        dat_cor = lapply(list_of_algos, function(x) unlist(lapply(x$PREDS, function(y) lapply(y, function(e) max.col(e$pred_te))), recursive = FALSE))
      }
    }

    dat = lapply(list_of_algos, function(x) unlist(lapply(x$PREDS, function(y) lapply(y, function(e) do.call(evaluation_metric, list(e$y_te, e$pred_te))))))
  }

  eval_algs = lapply(dat, mean)

  evaluate_algs = unlist(eval_algs)

  ind = t(combn(tmp_names, 2))

  out_vec = COR_LST = list()

  for (i in 1:dim(ind)[1]) {

    out_vec[[i]] = unlist(t.test(dat[[ind[i, 1]]], dat[[ind[i, 2]]], paired = T, conf.level = t.test.conf.int))[c(3:6)]

    if (!is.null(cor_test)) {

      if (regression) {

        tmp_regr_cor = lapply(1:length(dat_cor[[1]]), function(x) lapply(1:length(dat_cor[[ind[i, 1]]][[x]]), function(y) stats::cor.test(dat_cor[[ind[i, 1]]][[x]][[y]], dat_cor[[ind[i, 2]]][[x]][[y]], method = cor_test$method, na.action = "na.exclude")))

        COR_LST[[i]] = colMeans(do.call(rbind, lapply(tmp_regr_cor, function(x) do.call(rbind, lapply(x, function(e) unlist(list(e[names(e)[4]], e[names(e)[3]])))))))
      }

      else {

        tmp_COR = lapply(1:length(dat_cor[[1]]), function(x) stats::cor.test(dat_cor[[ind[i, 1]]][[x]], dat_cor[[ind[i, 2]]][[x]], method = cor_test$method, na.action = "na.exclude"))

        COR_LST[[i]] = colMeans(do.call(rbind, lapply(tmp_COR, function(x) unlist(list(x[names(x)[4]], x[names(x)[3]])))))
      }
    }
  }

  tmp_df = data.frame(do.call(rbind, out_vec), stringsAsFactors = F)
  for (j in 1:dim(tmp_df)[2]) {tmp_df[, j] = as.numeric(tmp_df[, j])}
  colnames(tmp_df) = paste('t.test', c('p.value', 'conf.int.min', 'conf.int.max', 'mean.of.diffs'), sep = '.')
  tmp_df = round(tmp_df, 4)

  if (!is.null(cor_test)) {

    tmp_out_cor = round(do.call(rbind, COR_LST), 4)
    tmp_out = cbind(ind, rep(' || ', dim(tmp_df)[1]), tmp_df, rep(' || ', dim(tmp_df)[1]), tmp_out_cor, rep(' || ', dim(tmp_df)[1]), evaluate_algs[match(ind[, 1], names(eval_algs))], evaluate_algs[match(ind[, 2], names(eval_algs))])
    colnames(tmp_out) = c('algorithm_1', 'algorithm_2', ' || ', colnames(tmp_df), ' || ', paste(cor_test$method, colnames(tmp_out_cor), sep = '_'), ' || ', paste(evaluation_metric, c('algorithm_1', 'algorithm_2'), sep = '_'))
    tmp_out[order(tmp_out[, dim(tmp_out)[2] - 1], decreasing = sort_decreasing), ]
  }

  else {

    tmp_out = cbind(ind, rep(' || ', dim(tmp_df)[1]), tmp_df, rep(' || ', dim(tmp_df)[1]), evaluate_algs[match(ind[, 1], names(eval_algs))], evaluate_algs[match(ind[, 2], names(eval_algs))])
    colnames(tmp_out) = c('algorithm_1', 'algorithm_2', ' || ', colnames(tmp_df), ' || ', paste(evaluation_metric, c('algorithm_1', 'algorithm_2'), sep = '_'))
    tmp_out[order(tmp_out[, dim(tmp_out)[2] - 1], decreasing = sort_decreasing), ]
  }
}

