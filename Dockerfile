FROM rocker/rstudio:devel 

 
LABEL maintainer='Lampros Mouselimis' 

 
RUN export DEBIAN_FRONTEND=noninteractive; apt-get -y update && \ 
 apt-get install -y openjdk-7-jre-headless make libcurl4-openssl-dev libgmp-dev libxml2-dev zlib1g-dev libssl-dev && \ 
 apt-get install -y sudo && \ 
 apt-get -y update && \ 
 R -e "install.packages(c( 'stats', 'utils', 'h2o', 'xgboost', 'elmNNRcpp', 'RWeka', 'gbm', 'kknn', 'MASS', 'nnet', 'e1071', 'LiblineaR', 'extraTrees', 'glmnet', 'randomForest', 'kernlab', 'caret', 'adabag', 'testthat', 'covr', 'ranger', 'remotes' ), repos =  'https://cloud.r-project.org/' )" && \ 
 R -e "remotes::install_github(c('mlampros/FeatureSelection'), upgrade = 'always', dependencies = TRUE, repos = 'https://cloud.r-project.org/')" && \ 
 R -e "remotes::install_github('mlampros/RandomSearchR', upgrade = 'always', dependencies = TRUE, repos = 'https://cloud.r-project.org/')" && \ 
 apt-get autoremove -y && \ 
 apt-get clean 

 
ENV USER rstudio 

 
