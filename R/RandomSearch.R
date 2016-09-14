
#' random_search_resample
#'
#' This function finds the optimal parameters of an algorithm using random search
#'
#' @param y a numeric vector
#' @param tune_iters a number
#' @param resampling_method one of 'bootstrap', 'train_test_split', 'cross_validation'
#' @param ALGORITHM a list of parameters
#' @param grid_params a grid of parameters in form of a list
#' @param DATA a list including the data
#' @param Args a list with further arguments of the function
#' @param regression a boolean (TRUE, FALSE)
#' @param re_run_params a boolean (TRUE, FALSE)
#' @return a list of lists
#' @author Lampros Mouselimis
#' @details
#' This function takes a number of arguments (including a grid of parameters) of an algorithm and using random search it returns a list of predictions and parameters for the chosen resampling method.
#' @export
#' @importFrom h2o h2o.init
#' @importFrom h2o as.h2o
#' @importFrom h2o h2o.predict
#' @importFrom h2o h2o.no_progress
#' @importFrom xgboost xgb.DMatrix
#' @importFrom xgboost getinfo
#' @importFrom stats as.formula
#' @importFrom utils txtProgressBar
#' @importFrom utils setTxtProgressBar
#' @import kknn
#' @examples
#'
#' # MULTICLASS CLASSIFICATION
#'
#' # library(kknn)
#' # data(glass)
#'
#' # str(glass)
#'
#' # X = glass[, -c(1, dim(glass)[2])]
#' # y1 = glass[, dim(glass)[2]]
#'
#' # form <- as.formula(paste('Type ~', paste(names(X),collapse = '+')))
#'
#' # y1 = c(1:length(unique(y1)))[ match(y1, sort(unique(y1))) ]       # labels should begin from 1:Inf
#' # ALL_DATA = glass
#' # ALL_DATA$Type = as.factor(y1)
#'
#' # randomForest classifier
#'
#' # wrap_grid_args3 = list(ntree = seq(30, 50, 5), mtry = c(2:3), nodesize = seq(5, 15, 5))
#'
#' # res_rf = random_search_resample(as.factor(y1), tune_iters = 15,
#' #
#' #                                resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
#' #
#' #                                ALGORITHM = list(package = require(randomForest), algorithm = randomForest),
#' #
#' #                                 grid_params = wrap_grid_args3,
#' #
#' #                               DATA = list(x = X, y = as.factor(y1)),
#' #
#' #                               Args = NULL,
#' #
#' #                               regression = FALSE, re_run_params = FALSE)
#'
#'
#' # Logit boost
#'
#' #[ RWeka::WOW("LogitBoost") : gives info for the parameters of the RWeka control list ]
#'
#' # lb_lst = list(control = RWeka::Weka_control(H = c(1.0, 0.5), I = seq(10, 30, 5), Q = c(TRUE, FALSE), O = 4))
#'
#'
#' # res_log_boost = random_search_resample(as.factor(y1), tune_iters = 15,
#'
#' #                                       resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
#'
#' #                                       ALGORITHM = list(package = require(RWeka), algorithm = LogitBoost),
#'
#' #                                       grid_params = lb_lst,
#'
#' #                                       DATA = list(formula = form, data = ALL_DATA),
#'
#' #                                       Args = NULL,
#'
#' #                                       regression = FALSE, re_run_params = FALSE)



random_search_resample = function(y, tune_iters = NULL, resampling_method = NULL, ALGORITHM = NULL, grid_params = NULL, DATA = NULL, Args = NULL, regression = FALSE, re_run_params = FALSE, ...) {

  if (any(c('h2o.x', 'h2o.y') %in% names(DATA))) {

    suppressMessages(ALGORITHM$package)
    localH2O <- h2o.init(...)}

  else {

    if (!is.null(ALGORITHM$package)) suppressMessages(ALGORITHM$package)
  }

  if (re_run_params == TRUE) tune_iters = max(length_grid(grid_params))

  if (is.null(resampling_method)) stop(simpleError('give list with resampling parameters'))
  if (is.null(resampling_method$repeats) && is.null(resampling_method$sample_rate) && is.null(resampling_method$folds)) stop(simpleError('choose a resampling method'))
  if ((resampling_method$repeats < 2 || is.null(resampling_method$sample_rate)) && is.null(resampling_method$folds)) stop(simpleError('the number of repeats should be at least 2 and the sample_rate should be a float between 0 and 1'))
  if ((resampling_method$folds < 2) && (is.null(resampling_method$repeats) && is.null(resampling_method$sample_rate))) stop(simpleError('the number of folds should be greater than 1'))
  if (is.null(tune_iters) || tune_iters < 1) stop(simpleError('give number of iterations, which should be greater or equal to 1'))
  if (resampling_method$method %in% c('bootstrap', 'train_test_split') && !is.null(resampling_method$folds)) stop(simpleError('folds should be NULL'))
  if (resampling_method$method == 'cross_validation' && (!is.null(resampling_method$repeats) || !is.null(resampling_method$sample_rate))) stop(simpleError('repeats and sample_rate should be NULL'))

  if ('control' %in% names(grid_params) && class(grid_params$control) == "Weka_control") {

    TMP_lst_names = names(grid_params$control)}

  else {

    TMP_lst_names = unlist(lapply(1:length(grid_params), function(x) if (is.list(grid_params[[x]])) names(grid_params[[x]]) else names(grid_params)[[x]]))
  }

  GRID_lst = Grid_params = list()

  cat('\n') ; cat('grid random search starts ..', '\n')

  pb <- txtProgressBar(min = 1, max = tune_iters, style = 3) ; cat('\n')

  for (iter in 1:tune_iters) {

    #----------------------------------------------------------------------------------------------------------------------------------------------------
    # In case I do change something in this condition THEN I have to change also the 2nd Condition [ continue downwards ]

    if ('target' %in% names(DATA) && 'data' %in% names(DATA)) {

      tmp_X = DATA[['data']]
      tmp_y = DATA[['target']]}

    if ('formula' %in% names(DATA) && 'data' %in% names(DATA)) {

      tmp_X = DATA[['data']][, all.vars(DATA[['formula']])]
      tmp_y = tmp_X[, 1]}

    if ('x' %in% names(DATA)) {

      tmp_y = DATA[['y']]
      tmp_X = DATA[['x']]}

    if ('train' %in% names(DATA) && 'formula' %in% names(DATA)) {

      tmp_X = DATA[['train']][, all.vars(DATA[['formula']])]
      tmp_y = tmp_X[, 1]
    }

    if ('data' %in% names(DATA) && 'y' %in% names(DATA)) {

      tmp_y = DATA[['y']]
      tmp_X = DATA[['data']]
    }

    if ('watchlist' %in% names(DATA))  {

      if (regression == TRUE) {

        tmp_y = DATA[['watchlist']][['label']]}

      else {

        tmp_y = DATA[['watchlist']][['label']] - 1            # subtract 1 as xgboost classification works from 0:Inf
      }

      tmp_X = DATA[['watchlist']][['data']]
    }

    if (any(c('h2o.x', 'h2o.y') %in% names(DATA))) {

      tmp_h2o_data = cbind(DATA[['h2o.y']], DATA[['h2o.x']])
    }

    # END of 1st exception for various algorithms
    #-----------------------------------------------------------------------------------------------------------------------------------------------------

    if (re_run_params == FALSE) {

      alg_fit = function_grid(grid_params)               # This function extracts a sample-set of parameters

      if (length(grep("Weka", class(ALGORITHM$algorithm) , perl = TRUE, value = FALSE) > 0)) {

        tmp_par_wek = as.data.frame(alg_fit$control[names(alg_fit$control)])

        for (i in 1:dim(tmp_par_wek)[2]) if (is.logical(tmp_par_wek[, i])) tmp_par_wek[, i] = as.character(tmp_par_wek[, i])

        Grid_params[[iter]] = tmp_par_wek}

      else {

        Grid_params[[iter]] = unlist(alg_fit)
      }
    }

    else {

      if (length(grep("Weka", class(ALGORITHM$algorithm) , perl = TRUE, value = FALSE) > 0)) {

        TMP_wek = func_weka_rerun(grid_params)

        alg_fit = TMP_wek[[iter]]

        Grid_params[[iter]] = do.call(cbind, grid_params)[iter, ]}

      else {

        alg_fit = optimal_grid(grid_params, iter)          # This function re-runs the optimal parameters

        Grid_params[[iter]] = unlist(alg_fit)
      }
    }


    out_resampling = resampling_methods(y, resampling_method$method, iter, REPEATS = resampling_method$repeats,

                                        sample_rate = resampling_method$sample_rate, FOLDS = resampling_method$folds)
    out_f = list()

    for (Repeat in 1:length(out_resampling$idx_train)) {

      idx_train = out_resampling$idx_train[[Repeat]]
      idx_test = out_resampling$idx_test[[Repeat]]

      #---------------------------------------------------------------------------------------------------------------------------------------------------
      # 2nd Condition: adjust here if I have changed the previous 1st condition [ otherwise error ]

      if ('x' %in% names(DATA)) {

        tmp_args = DATA

        for (i in names(alg_fit)) { tmp_args[[i]] = alg_fit[[i]] }

        tmp_args[['x']] = tmp_X[idx_train, ]

        tmp_args[['y']] = tmp_y[idx_train]

        if (!is.null(Args)) {

          for (i in names(Args)) { tmp_args[[i]] = Args[[i]] }
        }


        tmp_fit = do.call(ALGORITHM$algorithm, tmp_args)

        if (regression == TRUE) {

          pred_tr = as.vector(EXCEPTIONS_preds(tmp_fit, tmp_X[idx_train, ], regression))
          pred_te = as.vector(EXCEPTIONS_preds(tmp_fit, tmp_X[idx_test, ], regression))}

        else {

          pred_tr = EXCEPTIONS_preds(tmp_fit, tmp_X[idx_train, ], regression)
          pred_te = EXCEPTIONS_preds(tmp_fit, tmp_X[idx_test, ], regression)
        }

        out_f[[Repeat]] = list(pred_tr = pred_tr, pred_te = pred_te, y_tr = tmp_y[idx_train], y_te = tmp_y[idx_test])
      }

      if ('formula' %in% names(DATA) && 'data' %in% names(DATA)) {

        tmp_args = DATA

        if ('control' %in% names(alg_fit) && class(alg_fit$control) == "Weka_control") {                       # exception for RWeka, revert the assigned list

          tmp_args[['control']] =  alg_fit$control}

        else {

          for (i in names(alg_fit)) { tmp_args[[i]] = alg_fit[[i]] }
        }

        tmp_args[['data']] = tmp_X[idx_train, ]

        if (!is.null(Args)) {

          for (i in names(Args)) { tmp_args[[i]] = Args[[i]] }
        }

        tmp_fit = do.call(ALGORITHM$algorithm, tmp_args)

        if (regression == TRUE) {

          pred_tr = as.vector(EXCEPTIONS_preds(tmp_fit, tmp_X[idx_train, ], regression))
          pred_te = as.vector(EXCEPTIONS_preds(tmp_fit, tmp_X[idx_test, ], regression))}

        else {

          pred_tr = EXCEPTIONS_preds(tmp_fit, tmp_X[idx_train, ], regression)
          pred_te = EXCEPTIONS_preds(tmp_fit, tmp_X[idx_test, ], regression)
        }

        out_f[[Repeat]] = list(pred_tr = pred_tr, pred_te = pred_te, y_tr = tmp_y[idx_train], y_te = tmp_y[idx_test])
      }

      if ('target' %in% names(DATA) && 'data' %in% names(DATA)) {

        tmp_args = DATA

        for (i in names(alg_fit)) { tmp_args[[i]] = alg_fit[[i]] }

        tmp_args[['target']] = tmp_y[idx_train]

        tmp_args[['data']] = tmp_X[idx_train, ]

        if (!is.null(Args)) {

          for (i in names(Args)) { tmp_args[[i]] = Args[[i]] }
        }

        tmp_fit = do.call(ALGORITHM$algorithm, tmp_args)

        if (regression == TRUE) {

          pred_tr = as.vector(EXCEPTIONS_preds(tmp_fit, tmp_X[idx_train, ], regression))
          pred_te = as.vector(EXCEPTIONS_preds(tmp_fit, tmp_X[idx_test, ], regression))}

        else {

          pred_tr = EXCEPTIONS_preds(tmp_fit, tmp_X[idx_train, ], regression)
          pred_te = EXCEPTIONS_preds(tmp_fit, tmp_X[idx_test, ], regression)
        }

        out_f[[Repeat]] = list(pred_tr = pred_tr, pred_te = pred_te, y_tr = tmp_y[idx_train], y_te = tmp_y[idx_test])
      }

      if ('train' %in% names(DATA)) {

        tmp_args = DATA

        for (i in names(alg_fit)) { tmp_args[[i]] = alg_fit[[i]] }

        tmp_args[['train']] = tmp_X[idx_train, ]

        tmp_args[['test']] = tmp_X[idx_train, ]

        if (!is.null(Args)) {

          for (i in names(Args)) { tmp_args[[i]] = Args[[i]] }
        }

        tmp_fit_tr = do.call(ALGORITHM$algorithm, tmp_args)

        tmp_args[['test']] = tmp_X[idx_test, ]

        tmp_fit_te = do.call(ALGORITHM$algorithm, tmp_args)

        if (regression == TRUE) {

          tmp_out = list(pred_tr = tmp_fit_tr$fitted.values, pred_te = tmp_fit_te$fitted.values, y_tr = tmp_y[idx_train], y_te = tmp_y[idx_test])}

        else {

          tmp_out = list(pred_tr = tmp_fit_tr$prob, pred_te = tmp_fit_te$prob, y_tr = tmp_y[idx_train], y_te = tmp_y[idx_test])
        }

        out_f[[Repeat]] = tmp_out
      }

      if ('data' %in% names(DATA) && 'y' %in% names(DATA)) {

        tmp_args = alg_fit

        tmp_args[['y']] = tmp_y[idx_train]

        tmp_args[['data']] = tmp_X[idx_train, ]

        tmp_args[['TEST_data']] = NULL

        if (!is.null(Args)) {

          for (i in names(Args)) { tmp_args[[i]] = Args[[i]] }
        }

        tmp_fit_tr = do.call(ALGORITHM$algorithm, tmp_args)

        tmp_args[['TEST_data']] = tmp_X[idx_test, ]

        tmp_fit_te = do.call(ALGORITHM$algorithm, tmp_args)

        out_f[[Repeat]] = list(pred_tr = tmp_fit_tr, pred_te = tmp_fit_te, y_tr = tmp_y[idx_train], y_te = tmp_y[idx_test])
      }

      if (any(c('h2o.x', 'h2o.y') %in% names(DATA))) {                    # see if you can reduce the time when subsetting an h2o.frame as it takes time especially during resampling where I have to do multiple splits

        h2o.no_progress()                                                 # hide the progress bar when h2o runs

        tmp_args = alg_fit

        h2o_train = as.h2o(tmp_h2o_data[idx_train, ], destination_frame = 'h2o_train')
        h2o_test = as.h2o(tmp_h2o_data[idx_test, ], destination_frame = 'h2o_test')

        tmp_args[['training_frame']] = h2o_train

        tmp_args[['y']] = 1

        tmp_args[['x']] = 2:dim(tmp_h2o_data)[2]

        if (!is.null(Args)) {

          for (i in names(Args)) { tmp_args[[i]] = Args[[i]] }
        }

        tmp_fit_h2o = do.call(ALGORITHM$algorithm, tmp_args)

        if (regression == TRUE && as.vector(class(tmp_fit_h2o)) == "H2ORegressionModel") {

          pred_tr = as.data.frame(h2o.predict(object = tmp_fit_h2o, newdata = h2o_train))[, 1]
          pred_te = as.data.frame(h2o.predict(object = tmp_fit_h2o, newdata = h2o_test))[, 1]}

        else if (regression == FALSE && (as.vector(class(tmp_fit_h2o)) %in% c("H2OBinomialModel", "H2OMultinomialModel"))) {

          pred_tr = as.data.frame(h2o.predict(object = tmp_fit_h2o, newdata = h2o_train))[, -1]
          pred_te = as.data.frame(h2o.predict(object = tmp_fit_h2o, newdata = h2o_test))[, -1]}

        else {

          stop(simpleError("choose the right 'distribution' or 'family' for the data in case of regression or classification"))
        }

        out_f[[Repeat]] = list(pred_tr = pred_tr, pred_te = pred_te, y_tr = tmp_h2o_data[idx_train, 1], y_te = tmp_h2o_data[idx_test, 1])
      }

      if ('watchlist' %in% names(DATA)) {

        tmp_args = alg_fit

        if (is.matrix(tmp_X) || (class(tmp_X) == 'dgCMatrix')) tmp_args[['data']] = xgb.DMatrix(data = tmp_X[idx_train, ], label = tmp_y[idx_train], missing = NA) else tmp_args[['data']] = xgb.DMatrix(data = as.matrix(tmp_X[idx_train, ]), label = tmp_y[idx_train], missing = NA)

        if (is.matrix(tmp_X) || (class(tmp_X) == 'dgCMatrix')) tmp_args[['watchlist']] = list(train = xgb.DMatrix(data = tmp_X[idx_train, ], label = tmp_y[idx_train], missing = NA),

            test = xgb.DMatrix(data = tmp_X[idx_test, ], label = tmp_y[idx_test], missing = NA)) else

              tmp_args[['watchlist']] = list(train = xgb.DMatrix(data = as.matrix(tmp_X[idx_train, ]), label = tmp_y[idx_train], missing = NA),

                                             test = xgb.DMatrix(data = as.matrix(tmp_X[idx_test, ]), label = tmp_y[idx_test], missing = NA))

            if (!is.null(Args)) {

              for (i in names(Args)) { tmp_args[[i]] = Args[[i]] }
            }

            tmp_fit = do.call(ALGORITHM$algorithm, tmp_args)

            if (regression == TRUE) {

              pred_tr = xgboost::predict(tmp_fit, tmp_args$watchlist$train, ntreelimit = tmp_fit$bestInd)
              pred_te = xgboost::predict(tmp_fit, tmp_args$watchlist$test, ntreelimit = tmp_fit$bestInd)
              out_f[[Repeat]] = list(pred_tr = pred_tr, pred_te = pred_te, y_tr = tmp_y[idx_train], y_te = tmp_y[idx_test])}

            else {

              pred_tr = as.data.frame(matrix(xgboost::predict(tmp_fit, tmp_args$watchlist$train, ntreelimit = tmp_fit$bestInd), nrow = nrow(tmp_X[idx_train, ]), ncol = length(unique(getinfo(tmp_args$data, 'label'))), byrow = T))
              pred_te = as.data.frame(matrix(xgboost::predict(tmp_fit, tmp_args$watchlist$test, ntreelimit = tmp_fit$bestInd), nrow = nrow(tmp_X[idx_test, ]), ncol = length(unique(getinfo(tmp_args$data, 'label'))), byrow = T))
              out_f[[Repeat]] = list(pred_tr = pred_tr, pred_te = pred_te, y_tr = as.factor(tmp_y[idx_train] + 1), y_te = as.factor(tmp_y[idx_test] + 1))      # I subtracted 1 from tmp_y AND I add here 1 for the performance measures [ convert to factor in classification ]
                        }
      }

      # END of 2nd exception for various algorithms
      #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    }

    GRID_lst[[iter]] = out_f

    setTxtProgressBar(pb, iter)
  }
  close(pb) ; cat('\n')

  tmp_PARAMS = data.frame(do.call(rbind, Grid_params), stringsAsFactors = FALSE)

  colnames(tmp_PARAMS) = TMP_lst_names

  return(list(PREDS = GRID_lst, PARAMS = tmp_PARAMS))
}

