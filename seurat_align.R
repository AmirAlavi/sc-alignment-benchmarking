library(scater)
library(loomR)
library(Seurat)

args = commandArgs(trailingOnly=TRUE)

batch_key <-  #args[1] #"protocol"
n_dims <- 30 #as.integer(args[2])
print(n_dims)

counts <- read.csv(file = "_tmp_counts.csv", header = TRUE, sep = ",", row.names = 1, check.names = FALSE)
meta <- read.csv(file = "_tmp_meta.csv", header = TRUE, sep = ",", row.names = 1)
print(dim(counts))
# str(counts)
# rownames(counts)
# colnames(counts)
# str(meta)
# rownames(meta)
# print(colnames(counts)[1])
# print(rownames(meta)[1])
#print(setdiff(x = rownames(x = meta), y = colnames(x = counts)))

data <- CreateSeuratObject(counts, meta.data=meta)
print("created seurat object")
data.list <- SplitObject(data, split.by = batch_key)
print("split seurat object")

print("normalizing data...")
for (i in 1:length(data.list)) {
    print(i)
    data.list[[i]] <- NormalizeData(data.list[[i]], verbose = FALSE)
    print("normalized")
    data.list[[i]] <- FindVariableFeatures(data.list[[i]], selection.method = "vst", 
        nfeatures = 2000, verbose = FALSE)
    print("found variable features")
}

reference.list <- data.list
print("Finding anchors...")
data.anchors <- FindIntegrationAnchors(object.list = reference.list, dims = 1:n_dims)
print("done")

data.integrated <- IntegrateData(anchorset = data.anchors, dims = 1:n_dims)
DefaultAssay(data.integrated) <- "integrated"

print("data.integrated shape")
print(dim(data.integrated))
print(str(data.integrated))


data.integrated <- FindVariableFeatures(object = data)
print("after FindVariableFeatures")
print(str(data.integrated))
fn <- "_tmp_adata_for_seurat.loom"
if (file.exists(fn)) {
    print("old loom file exists, deleting")
    file.remove(fn)
}
print(dim(data.integrated))
data.loom <- as.loom(data.integrated, filename = fn, verbose = FALSE)
print(str(data.loom))
data.loom$close_all()