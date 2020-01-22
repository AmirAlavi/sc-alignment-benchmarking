library(scater)
library(loomR)
library(Seurat)

args = commandArgs(trailingOnly=TRUE)

batch_key <- args[1] #"protocol"
n_dims <- as.integer(args[2])
count_file <- args[3]
meta_file <- args[4]
result_file <- args[5]
print(n_dims)

counts <- read.csv(file = count_file, header = TRUE, sep = ",", row.names = 1, check.names = FALSE)
print("head(counts, n = 5)")
print(head(counts, n = 5))
meta <- read.csv(file = meta_file, header = TRUE, sep = ",", row.names = 1)
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

n_features <- min(dim(counts)[1], 2000)

print("normalizing data...")
for (i in 1:length(data.list)) {
    print(i)
    data.list[[i]] <- NormalizeData(data.list[[i]], verbose = TRUE)
    print("normalized")
    data.list[[i]] <- FindVariableFeatures(data.list[[i]], selection.method = "vst", 
        nfeatures = n_features, verbose = TRUE)
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
print("head(data.integrated, n = 5)")
print(head(data.integrated, n = 5))

 
## data.integrated <- FindVariableFeatures(object = data)
## print("after FindVariableFeatures")
## print(dim(data.integrated))
fn <- result_file
if (file.exists(fn)) {
    print("old loom file exists, deleting")
    file.remove(fn)
}
print(dim(data.integrated))
data.loom <- as.loom(data.integrated, filename = fn, verbose = FALSE)
data.loom$close_all()
