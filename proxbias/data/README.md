## Data Sources

### Gene annotations
`centromeres_hg38.tsv`, `hg38_cytoband.tsv.gz`, `hg38_scaffolds.tsv`, `ncbirefseq_hg38.tsv.gz` were downloaded from UCSC https://genome.ucsc.edu/cgi-bin/hgTables

### shinyDepMap_19Q3_cluster_s.csv
This inferred gene-gene connectivity csv file was processed from [shinyDepMap](https://github.com/kenichi-shimada/shinyDepMap) repo using the following R code. To ensure that the analysis is consistent with the default behavior of shinyDepMap, we used the `small` gene clusters inferred by CRISPR-only dataset (no shRNA).
```R
# https://labsyspharm.shinyapps.io/depmap/
# Download data file if not present in the current directory.
filename <- "depmap_initial_19q3_v3_local_run.rda"
if (!file.exists(filename) | file.size(filename) != 415441828) {
  cat("\nDownloading data file...\n\n")
  options(timeout = .Machine$integer.max)
  if (download.file("https://ndownloader.figshare.com/files/25893237", filename, mode = "wb")) {
    file.remove(filename)
    stop("Could not download data file. Please check your network connection and try again.")
  } else {
    cat("\nDownload successful\n\n")
  }
}
# Load the data file, which should exist at this point.
cat("Loading data file...\n")
x <- load(filename)

sym <- com$sym
eid <- com$eid
crispr.q1 <- dfs$CRISPR$q1
eid2sym <- sym
names(eid2sym) <- eid

# CRISPR gene efficacy & selectivity data
gene_efficacy_selectivity <- merge(data.frame(sym, eid), crispr.q1)

# CRISPR gene clusters
# nodes edges info
cluster_s <- do.call(rbind, lapply(graphs$CRISPR$mem1, function(x) x$edges))
cluster_s$from <- eid2sym[cluster_s$from]
cluster_s$to <- eid2sym[cluster_s$to]

# Save to csv
write.csv(cluster_s, file="shinyDepMap_19Q3_cluster_s.csv", row.names=FALSE)
write.csv(gene_efficacy_selectivity, file="shinyDepMap_19Q3_efficacy_selectivity.csv", row.names=FALSE)
```