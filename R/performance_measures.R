
#' secondary function used in performance measures   [ order of calculation is : first 'y_true' then 'preds' ]
#' 
#' @keywords internal

second_preds = function(sublist, eval_metric) {

  metr_tr = do.call(eval_metric, list(sublist[['y_tr']], sublist[['pred_tr']]))
  metr_te = do.call(eval_metric, list(sublist[['y_te']], sublist[['pred_te']]))

  list(tr = metr_tr, te = metr_te)
}



#' performance_measures
#'
#' This function takes a list of objects of the random_search_resample function and returns a list with the optimal parameters

#' @param list_objects a list of model objects
#' @param eval_metric the evaluation metric (the name of the function)
#' @param sort a list of arguments specifying how the optimal parameters should be sorted (for variable one of : 'Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.')
#' @return a list of lists
#' 
#' @importFrom stats complete.cases
#' 
#' @details
#' This function takes a list of objects of the random_search_resample function and returns a list with the optimal parameters. Four lists are returned : the first is a list of the grid parameters
#' evaluated on the train data, the second is the same list of parameters evaluated on test data, the third gives summary statistics for the predictions of each algorithm compared with the other
#' algorithms and the fourth list shows if any of the models had missing values in the predictions.
#' 
#' @export
#' 
#' @examples
#'
#' \dontrun{
#' 
#' perf = performance_measures(list_objects = list(rf = res_rf, logitBoost = res_log_boost), eval_metric = acc, sort = list(variable = 'Median', decreasing = TRUE))
#' 
#' perf
#' }

performance_measures = function(list_objects, eval_metric, sort = list(variable = 'Median', decreasing = TRUE)) {

  if (is.null(list_objects)) stop(simpleError("the list of objects is empty"))

  PARAMS_colnams = lapply(list_objects, function(y) colnames(y$PARAMS))

  NAs = lapply(list_objects, function(x) length(which(is.na(as.vector(unlist(x[['PREDS']]))))))

  tmp_params = list()

  for (i in names(list_objects)) {

    tmp_params[[i]] = lapply(1:length(list_objects[[i]][['PREDS']]), function(x) do.call(rbind, lapply(list_objects[[i]][['PREDS']][[x]], function(y) unlist(second_preds(y, eval_metric)))))
  }

  complete_cases_tr = lapply(tmp_params, function(x) stats::complete.cases(t(do.call(cbind, lapply(x, function(y) y[, 1])))))
  complete_cases_te = lapply(tmp_params, function(x) stats::complete.cases(t(do.call(cbind, lapply(x, function(y) y[, 2])))))

  tmp_complete = lapply(lapply(1:length(complete_cases_tr), function(x) data.frame(tr = complete_cases_tr[[x]], te = complete_cases_te[[x]])), function(f) apply(f, 1, all))   # remove cases where both train and test predictions include NAs

  for (obj in 1:length(list_objects)) {

    list_objects[[names(list_objects)[obj]]]$PREDS = list_objects[[names(list_objects)[obj]]]$PREDS[which(tmp_complete[[obj]] == T)]
    list_objects[[names(list_objects)[obj]]]$PARAMS = list_objects[[names(list_objects)[obj]]]$PARAMS[which(tmp_complete[[obj]] == T), ]
  }

  tmp_params1 = list()

  for (i in names(list_objects)) {

    tmp_params1[[i]] = lapply(1:length(list_objects[[i]][['PREDS']]), function(x) do.call(rbind, lapply(list_objects[[i]][['PREDS']][[x]], function(y) unlist(second_preds(y, eval_metric)))))
  }

  tmp_lst_tr = lapply(tmp_params1, function(x) t(apply(do.call(cbind, lapply(x, function(y) y[, 1])), 2, summary)))
  tmp_lst_te = lapply(tmp_params1, function(x) t(apply(do.call(cbind, lapply(x, function(y) y[, 2])), 2, summary)))

  func_PARAMS = function(dat, Colnames) {

    colnames(dat) = Colnames
    return(dat)
  }

  PARAMS = list()

  for (param in 1:length(PARAMS_colnams)) {

    PARAMS[[param]] = func_PARAMS(data.frame(list_objects[[param]]$PARAMS), PARAMS_colnams[[param]])
  }

  out_tr = out_te = list()

  for (f in 1:length(tmp_lst_tr)) {

    out_tr[[names(list_objects[f])]] = cbind(PARAMS[[f]], tmp_lst_tr[[f]])
    out_te[[names(list_objects[f])]] = cbind(PARAMS[[f]], tmp_lst_te[[f]])
  }

  out_tr_sort = lapply(out_tr, function(x) x[order(x[, sort$variable], decreasing = sort$decreasing), ])
  out_te_sort = lapply(out_te, function(x) x[order(x[, sort$variable], decreasing = sort$decreasing), ])

  list(train_params = out_tr_sort, test_params = out_te_sort, train_resampling = do.call(rbind, lapply(tmp_lst_tr, function(x) apply(x, 2, mean))),

       test_resampling = do.call(rbind, lapply(tmp_lst_te, function(x) apply(x, 2, mean))), NAs_in_predictions = unlist(NAs))
}


