library(devtools)
devtools::install_github('satijalab/seurat-data')

library(BiocManager)
BiocManager::install("scater")

library(remotes)
remotes::install_github(repo = 'mojaveazure/loomR', ref = 'develop')
