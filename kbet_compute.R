library(kBET)

args = commandArgs(trailingOnly=TRUE)

x <- read.csv(args[1], sep=",", header=FALSE)
#x <- read.csv('testing_kBET/x.csv', sep=",", header=FALSE)
batch <- read.csv(args[2], sep=",", header=FALSE)
#batch <- read.csv('testing_kBET/batch.csv', sep=",", header=FALSE)
result <- kBET(x, batch[['V1']], plot = FALSE)
stats <- data.frame(result$stats)
write.csv(stats, file = args[3])
