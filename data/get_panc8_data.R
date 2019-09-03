library(SeuratData)

data("panc8")

dir.create("panc8")

all_counts <- panc8@assays$RNA@counts
print(dim(all_counts))
batches <- unique(panc8@meta.data$dataset)


for (batch in batches) {
	print(batch)

	# get meta data for this batch
	indexer <- panc8@meta.data$dataset == batch

	counts <- all_counts[, indexer]
	counts <- as.data.frame(as.matrix(counts))
	print(dim(counts))
	meta.data <- panc8@meta.data
	meta.data <- meta.data[indexer, ]
	print(dim(meta.data))

	write.csv(counts, file = paste0("panc8_2", "/", batch, "_counts.csv"))
	write.csv(meta.data, file = paste0("panc8_2", "/", batch, "_meta.csv"))
}