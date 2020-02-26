library(SeuratData)
InstallData("pbmcsca")
data("pbmcsca")

dir.create("pbmcsca")

all_counts <- pbmcsca@assays$RNA@counts
print(dim(all_counts))
batches <- unique(pbmcsca@meta.data$Method)


for (batch in batches) {
	print(batch)

	# get meta data for this batch
	indexer <- pbmcsca@meta.data$Method == batch

	counts <- all_counts[, indexer]
	counts <- as.data.frame(as.matrix(counts))
	print(dim(counts))
	meta.data <- pbmcsca@meta.data
	meta.data <- meta.data[indexer, ]
	print(dim(meta.data))

	write.csv(counts, file = paste0("pbmcsca", "/", batch, "_counts.csv"))
	write.csv(meta.data, file = paste0("pbmcsca", "/", batch, "_meta.csv"))
}