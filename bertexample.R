#Source: https://blogs.rstudio.com/ai/posts/2019-09-30-bert-r/

Sys.setenv(TF_KERAS=1) 
# make sure we use python 3
reticulate::use_python('C:/Users/turgut.abdullayev/AppData/Local/Continuum/anaconda3/python.exe',
                       required=T)
# to see python version
reticulate::py_config()
