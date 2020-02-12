library(BiocManager)
Sys.setenv(R_INSTALL_STAGED = FALSE)
Sys.setenv(TAR = "/bin/tar")
BiocManager::install('multtest')
install.packages('Seurat')

library(devtools)
devtools::install_github('satijalab/seurat-data')

library(BiocManager)
BiocManager::install("scater")

library(remotes)
remotes::install_github(repo = 'mojaveazure/loomR', ref = 'develop')