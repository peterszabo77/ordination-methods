library(MASS)
library(Rtsne)
library(dplyr)

df = read.csv("irisdata.csv")
names(df) = c('sl', 'sw', 'pl', 'pw', 'label')

df$label = as.factor(df$label)

labels = sort(unique(df$label))
colorset = rainbow(length(labels))

names(colorset) = labels

labelcolors = colorset[df$label]

df_values = subset(df, select = -c(label))

# copy with no duplicate elements (for NMDS and tSE)
isduplicate = duplicated(df_values)
df_noduplicates = df[!isduplicate,]
df_noduplicates_values = subset(df_noduplicates, select = -c(label))
labelcolors_noduplicates = colorset[df_noduplicates$label]

# executing PCA
print('calculating PCA')
pca_result = prcomp(df_values, scale. = FALSE)
pca_scores = pca_result$x[,1:2]
PCA_x_scores = pca_scores[,1]
PCA_y_scores = pca_scores[,2]

# executing MDS
print('calculating MDS')
dissimilarities = dist(df_values, method = "euclidean")
mds_result = cmdscale(dissimilarities, eig=TRUE, k=2)
MDS_x = mds_result$points[,1]
MDS_y = mds_result$points[,2]

# executing NMDS
print('calculating NMDS')
dissimilarities = dist(df_noduplicates_values, method = "euclidean")
nmds_result = isoMDS(dissimilarities, k=2)
NMDS_x = nmds_result$points[,1]
NMDS_y = nmds_result$points[,2]

# Executing tSNE
print('calculating tSNE')
dissimilarities = dist(df_noduplicates_values, method = "euclidean")
tsne_result = Rtsne(dissimilarities, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
tSNE_x_scores = tsne_result$Y[,1]
tSNE_y_scores = tsne_result$Y[,2]

# Plotting
pdf(file='ordination_in_R.pdf', width = 16, height = 4)
par(mfrow=c(1,4))
plot(PCA_x_scores, PCA_y_scores, t='p', main="PCA", xlab="x", ylab="y", pch=16, col=labelcolors)
legend("topleft", legend=labels, col=colorset, lty=1, cex=0.8, bg='lightblue')
plot(MDS_x, MDS_y, t='p', main="MDS", xlab="x", ylab="y", pch=16, col=labelcolors)
plot(NMDS_x, NMDS_y, t='p', main="NMDS", xlab="x", ylab="y", pch=16, col=labelcolors_noduplicates)
plot(tSNE_x_scores, tSNE_y_scores, t='p', main="t-SNE", xlab="x", ylab="y", pch=16, col=labelcolors_noduplicates)
dev.off()
