library(scater)
library(loomR)
library(Seurat)

args = commandArgs(trailingOnly=TRUE)

batch_key <- args[1]

counts <- read.csv(file = "_tmp_counts.csv", header = TRUE, sep = ",", row.names = 1)
meta <- read.csv(file = "_tmp_meta.csv", header = TRUE, sep = ",", row.names = 1)
data <- CreateSeuratObject(counts, meta.data=meta)

data.list <- SplitObject(data, split.by = batch_key)

for (i in 1:length(data.list)) {
    data.list[[i]] <- NormalizeData(data.list[[i]], verbose = FALSE)
    data.list[[i]] <- FindVariableFeatures(data.list[[i]], selection.method = "vst", 
        nfeatures = 2000, verbose = FALSE)
}

reference.list <- data.list
data.anchors <- FindIntegrationAnchors(object.list = reference.list, dims = 1:30)

data.integrated <- IntegrateData(anchorset = data.anchors, dims = 1:30)
DefaultAssay(data.integrated) <- "integrated"


data.integrated <- FindVariableFeatures(object = data)
fn <- "_tmp_adata_for_seurat.loom"
if (file.exists(fn)) {
    file.remove(fn)
}
data.loom <- as.loom(data.integrated, filename = fn, verbose = FALSE)
data.loom$close_all()