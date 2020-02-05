
#' subset_mods
#'
#' subset the parameter output of a performance_measures object

#' @param perf_meas_OBJ a resulted object of the performance_measures function
#' @param bst_mods the number of models to select (rows of each algorithm)
#' @param train_params if the subset should be based on train or test output
#' @return a list of lists
#' 
#' @details
#' This function takes the result from the performance_measures function, the number of best performing grid-models and a boolean specifying if the train or test predictions should be taken into account
#' when subseting the data.frames and it returns a list with the optimal parameters for each algorithm.
#' 
#' @export
#' 
#' @examples
#'
#' \dontrun{
#' 
#' bst_m = subset_mods(perf_meas_OBJ = perf, bst_mods = 5, train_params = FALSE)
#' 
#' bst_m
#' }


subset_mods = function(perf_meas_OBJ, bst_mods = 5, train_params = FALSE) {

  if (bst_mods > min(unlist(lapply(perf_meas_OBJ$train_params, function(x) dim(x)[1])))) stop(simpleError("the number of 'bst_mods' should be less than or equal to the number of 'tune_iters' of each algorithm-model"))

  tmp_NAMS = names(perf_meas_OBJ$train_params)

  if (train_params) {

    tmp_colnams = lapply(perf_meas_OBJ$train_params, function(x) colnames(x)[-which(colnames(x) %in% c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max."))])   # take the column names in form of a list
    OUT = lapply(1:length(perf_meas_OBJ$train_params), function(x) as.data.frame(perf_meas_OBJ$train_params[[x]][1:bst_mods, tmp_colnams[[x]]]))
  }
  else {
    
    tmp_colnams = lapply(perf_meas_OBJ$test_params, function(x) colnames(x)[-which(colnames(x) %in% c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max."))])
    OUT = lapply(1:length(perf_meas_OBJ$test_params), function(x) as.data.frame(perf_meas_OBJ$test_params[[x]][1:bst_mods, tmp_colnams[[x]]]))
  }

  idx_lst_numeric = suppressWarnings(lapply(OUT, function(x) which(as.vector(colSums(is.na(apply(as.matrix(x), 2, as.numeric)))) == 0)))

  for (i in 1:length(idx_lst_numeric)) {                 # loop over the OUT list select each data.frame and change type of numeric parameters [ they were converted to character in the 'random_search_resample' function ]

    if (!is.null(idx_lst_numeric[[i]])) {

      for (j in 1:dim(OUT[[i]])[2]) {

        if (j %in% idx_lst_numeric[[i]]) {

          OUT[[i]][, j] = as.numeric(OUT[[i]][, j])
        }
      }
    }
  }

  # give the correct column-names
  for (k in 1:length(tmp_colnams)) {

    colnames(OUT[[k]]) = tmp_colnams[[k]]
  }

  # return list of lists with the parameters

  OUT_all = list()

  for (h in 1:length(OUT)) {

    tmp_lst = list()

    for (t in 1:dim(OUT[[h]])[2]) {

      tmp_lst[[colnames(OUT[[h]])[t]]] = OUT[[h]][, t]
    }

    OUT_all[[tmp_NAMS[h]]] = tmp_lst
  }

  OUT_all

}
