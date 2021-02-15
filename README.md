
[![tic](https://github.com/mlampros/RandomSearchR/workflows/tic/badge.svg?branch=master)](https://github.com/mlampros/RandomSearchR/actions)
[![codecov.io](https://codecov.io/github/mlampros/RandomSearchR/coverage.svg?branch=master)](https://codecov.io/github/mlampros/RandomSearchR?branch=master)
<a href="https://www.buymeacoffee.com/VY0x8snyh" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" height="21px" ></a>
[![](https://img.shields.io/docker/automated/mlampros/randomsearchr.svg)](https://hub.docker.com/r/mlampros/randomsearchr)

<br>


### Random search in R

<br>

The functions in the package can be used to:

* find the optimal parameters of an algorithm using random search
* compare performance of algorithms using summary statistics. 

<br>
To install the package from Github use,
<br><br>

```R
remotes::install_github('mlampros/RandomSearchR')

```
<br>

Once the package is downloaded use ? to read info about each function (i.e. ?random_search_resample). More details can be found in the [blog post](http://mlampros.github.io/2016/03/14/random_search_R/).
<br><br>

**UPDATE 05-02-2020**

<br>

**Docker images** of the *RandomSearchR* package are available to download from my [dockerhub](https://hub.docker.com/r/mlampros/randomsearchr) account. The images come with *Rstudio* and the *R-development* version (latest) installed. The whole process was tested on Ubuntu 18.04. To **pull** & **run** the image do the following,

<br>

```R

docker pull mlampros/randomsearchr:rstudiodev

docker run -d --name rstudio_dev -e USER=rstudio -e PASSWORD=give_here_your_password --rm -p 8787:8787 mlampros/randomsearchr:rstudiodev

```

<br>

The user can also **bind** a home directory / folder to the image to use its files by specifying the **-v** command,

<br>

```R

docker run -d --name rstudio_dev -e USER=rstudio -e PASSWORD=give_here_your_password --rm -p 8787:8787 -v /home/YOUR_DIR:/home/rstudio/YOUR_DIR mlampros/randomsearchr:rstudiodev


```

<br>

In the latter case you might have first give permission privileges for write access to **YOUR_DIR** directory (not necessarily) using,

<br>

```R

chmod -R 777 /home/YOUR_DIR


```

<br>

The **USER** defaults to *rstudio* but you have to give your **PASSWORD** of preference (see [www.rocker-project.org](https://www.rocker-project.org/) for more information).

<br>

Open your web-browser and depending where the docker image was *build / run* give, 

<br>

**1st. Option** on your personal computer,

<br>

```R
http://0.0.0.0:8787 

```

<br>

**2nd. Option** on a cloud instance, 

<br>

```R
http://Public DNS:8787

```

<br>

to access the Rstudio console in order to give your username and password.

<br>


