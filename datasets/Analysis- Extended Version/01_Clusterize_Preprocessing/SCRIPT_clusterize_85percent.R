
# ==========================================
# COMPLETE WORKFLOW: CSV TO FILTERED DATASET USING DECIPHER CLUSTERIZE
# ==========================================
# 
# OVERVIEW: This script uses DECIPHER's Clusterize function to reduce sequence 
# redundancy in a protein dataset while preserving functional diversity.
# 
# CLUSTERIZE ALGORITHM (3-PHASE PROCESS):
# Phase 1: Partition sequences into similarity groups using k-mer analysis
# Phase 2: Order sequences by k-mer similarity using relatedness sorting  
# Phase 3: Iteratively cluster sequences based on distance cutoff
# ==========================================

library(readr)        # For CSV file operations
library(dplyr)        # For data manipulation
library(DECIPHER)     # For Clusterize function
library(Biostrings)   # For sequence handling (AAStringSet)

# ==========================================
# STEP 1: DATA LOADING AND PREPARATION
# ==========================================

## Load the original enzyme dataset from CSV ## This is name of the data set provided to us by "reviewer 3" ##
## n = "27600" protein sequences from Uniprot from Reviewer 3.  After Clusterize analysis - we will obtain 20244 sequences.

cat("Loading original dataset...\n")

# this is the original_file_name

# df <- read_csv("uniprotkb_taxonomy_id_2759_AND_ec_AND_r_2025_06_27.csv") # original file name from Reviewer 3

# 9/16/25 changed file name to "INPUT_uniprot_27600_proteins.csv" to enable indepedent review.

df <- read_csv("INPUT_uniprot_27600_proteins.csv") 
cat("Original dataset:", nrow(df), "sequences\n")

## Create unique sequence identifiers by combining Entry and Organism
## This ensures each sequence has a unique name for tracking through clustering
df$SeqID <- paste(df$Entry, df$Organism, sep="_")

## Convert protein sequences to Biostrings AAStringSet format
## This is required for DECIPHER clustering functions
## AAStringSet is optimized for amino acid sequence operations
proteins <- AAStringSet(df$Sequence)
names(proteins) <- df$SeqID

# ==========================================
# STEP 2: DECIPHER CLUSTERIZE EXECUTION
# ==========================================

cat("Starting DECIPHER clustering...\n")
start_time <- Sys.time()

## CLUSTERIZE PARAMETERS EXPLANATION:
## cutoff = 0.3: Sequences with >70% identity will be clustered together
##               (distance of 0.3 = 30% differences = 70% identity)
## processors = 4: Use 4 CPU cores for parallel processing
## verbose = TRUE: Print progress updates during clustering
##
## WHAT CLUSTERIZE DOES:
## 1. PHASE 1: Groups sequences by shared k-mers (short sequence fragments)
##    - Identifies sequences likely to be similar based on k-mer content
##    - Creates initial partitions to reduce comparison space
##
## 2. PHASE 2: Orders sequences by relatedness using radix sorting
##    - Arranges sequences so similar ones are positioned near each other
##    - Uses k-mer similarity for efficient linear-time sorting
##
## 3. PHASE 3: Sequential clustering with distance cutoff
##    - First sequence becomes representative of cluster #1
##    - Each subsequent sequence is compared to existing cluster representatives
##    - If within cutoff distance (≤30% different), joins existing cluster
##    - If beyond cutoff distance, becomes new cluster representative
##    - Process continues until all sequences are assigned to clusters

clusters <- Clusterize(proteins, 
                       cutoff = 0.15,        # 70% identity threshold # adjusted to 85% to see and keep more sequences
                       processors = 4,       # Multi-core processing
                       verbose = TRUE)       # Show progress

end_time <- Sys.time()
cat("Clustering completed in", round(difftime(end_time, start_time, units="mins"), 2), "minutes\n")

# ==========================================
# STEP 3: CLUSTER REPRESENTATIVE SELECTION
# ==========================================

## UNDERSTANDING CLUSTER OUTPUT:
## clusters$cluster is a vector where each element indicates which cluster 
## each sequence belongs to (e.g., c(1,1,2,3,2,1) means sequences 1,2,6 
## are in cluster 1, sequences 3,5 are in cluster 2, sequence 4 is in cluster 3)

## Select one representative sequence from each cluster
## Strategy: Take the first sequence encountered in each cluster
## This preserves the ordering from the relatedness sorting in Phase 2
representatives_indices <- sapply(1:max(clusters$cluster), function(i) {
  cluster_members <- which(clusters$cluster == i)  # Find all sequences in cluster i
  cluster_members[1]  # Select first member (often most "central" due to sorting)
})

## Extract representative sequences using selected indices
representatives <- proteins[representatives_indices]
cat("Representatives selected:", length(representatives), "sequences\n")

# ==========================================
# STEP 4: COMPREHENSIVE FILTERING RESULTS TABLE
# ==========================================

## Create detailed tracking table for transparency and analysis
## This table allows users to understand clustering decisions
filtering_results <- data.frame(
  SeqID = names(proteins),                    # Unique sequence identifier
  OriginalIndex = 1:length(proteins),        # Position in original dataset
  ClusterID = clusters$cluster,              # Which cluster each sequence belongs to
  ClusterSize = sapply(clusters$cluster, function(x) sum(clusters$cluster == x)), # Size of each cluster
  IsRepresentative = 1:length(proteins) %in% representatives_indices, # TRUE/FALSE for representatives
  stringsAsFactors = FALSE
)

# ==========================================
# STEP 5: FILTER ORIGINAL DATASET
# ==========================================

## Create filtered dataset containing only cluster representatives
## This dramatically reduces redundancy while preserving diversity
filtered_df <- df %>%
  filter(SeqID %in% names(representatives))

# ==========================================
# STEP 6: CALCULATE AND DISPLAY STATISTICS
# ==========================================

## Quantify the impact of clustering-based filtering
original_count <- nrow(df)
filtered_count <- nrow(filtered_df)
reduction_percent <- round((1 - filtered_count/original_count) * 100, 1)

cat("\n=== FILTERING RESULTS ===\n")
cat("Original sequences:", original_count, "\n")
cat("Filtered sequences:", filtered_count, "\n")
cat("Reduction:", reduction_percent, "%\n")
cat("Sequences removed:", original_count - filtered_count, "\n")

# ==========================================
# STEP 7: ANALYZE CLUSTER SIZE DISTRIBUTION
# ==========================================

## Understanding which protein families are most redundant
## Large clusters indicate highly similar sequences (potential duplicates/close homologs)
cluster_summary <- filtering_results %>%
  group_by(ClusterID) %>%
  summarize(
    ClusterSize = n(),           # Count sequences per cluster
    .groups = 'drop'
  ) %>%
  arrange(desc(ClusterSize))     # Sort by cluster size (largest first)

cat("\nTop 10 largest clusters (most redundant families):\n")
print(head(cluster_summary, 10))

# ==========================================
# STEP 8: SAVE ALL OUTPUTS FOR TRANSPARENCY
# ==========================================

cat("\nSaving results...\n")

## Main result: filtered dataset with reduced redundancy
write_csv(filtered_df, "filtered_enzyme_dataset.csv")
cat("Saved: filtered_enzyme_dataset.csv\n")

## Transparency file: shows clustering assignments for every sequence
write_csv(filtering_results, "clustering_assignments.csv")
cat("Saved: clustering_assignments.csv\n")

## Summary statistics: cluster size distribution
write_csv(cluster_summary, "cluster_summary.csv")
cat("Saved: cluster_summary.csv\n")

## FASTA format: representative sequences for downstream analysis
writeXStringSet(representatives, "representative_sequences.fasta")
cat("Saved: representative_sequences.fasta\n")

# ==========================================
# STEP 9: QUALITY CONTROL CHECKS
# ==========================================

cat("\n=== QUALITY CONTROL ===\n")

## Check preservation of EC number diversity
## Important: Ensure clustering doesn't eliminate important enzyme classes
ec_original <- table(df$`EC number`)
ec_filtered <- table(filtered_df$`EC number`)

cat("EC number preservation check:\n")
ec_comparison <- data.frame(
  EC_Number = names(ec_original),
  Original_Count = as.numeric(ec_original),
  Filtered_Count = as.numeric(ec_filtered[names(ec_original)]),
  Retention_Percent = round(as.numeric(ec_filtered[names(ec_original)]) / as.numeric(ec_original) * 100, 1)
)
ec_comparison$Retention_Percent[is.na(ec_comparison$Retention_Percent)] <- 0

write_csv(ec_comparison, "ec_number_comparison.csv")
print(head(ec_comparison[order(-ec_comparison$Original_Count), ], 10))

## Check organism diversity preservation
## Important: Ensure clustering maintains taxonomic representation
organism_original <- length(unique(df$Organism))
organism_filtered <- length(unique(filtered_df$Organism))
cat("\nOrganism diversity:\n")
cat("Original:", organism_original, "unique organisms\n")
cat("Filtered:", organism_filtered, "unique organisms\n")
cat("Retention:", round(organism_filtered/organism_original * 100, 1), "%\n")

# ==========================================
# STEP 10: COMPREHENSIVE SUMMARY REPORT
# ==========================================

## Create executive summary of filtering impact
summary_report <- data.frame(
  Metric = c("Total Sequences", "Unique Organisms", "Unique EC Numbers", 
             "Avg Sequence Length", "Reduction Percentage"),
  Original = c(nrow(df), length(unique(df$Organism)), 
               length(unique(df$`EC number`)), round(mean(df$Length)), "-"),
  Filtered = c(nrow(filtered_df), length(unique(filtered_df$Organism)),
               length(unique(filtered_df$`EC number`)), 
               round(mean(filtered_df$Length)), paste0(reduction_percent, "%")),
  stringsAsFactors = FALSE
)

write_csv(summary_report, "filtering_summary_report.csv")
cat("\nFinal summary saved to: filtering_summary_report.csv\n")

# ==========================================
# WORKFLOW COMPLETION SUMMARY
# ==========================================

cat("\n=== WORKFLOW COMPLETE ===\n")
cat("Your filtered dataset is ready for downstream analysis!\n")
cat("Key files created:\n")
cat("  - filtered_enzyme_dataset.csv (main result)\n")
cat("  - clustering_assignments.csv (transparency)\n")
cat("  - filtering_summary_report.csv (overview)\n")

## KEY OUTPUTS EXPLAINED:
## 1. filtered_enzyme_dataset.csv: Reduced dataset with one representative per cluster
## 2. clustering_assignments.csv: Shows which sequences were clustered together
## 3. cluster_summary.csv: Statistics on cluster sizes (redundancy levels)
## 4. representative_sequences.fasta: Sequences in FASTA format for further analysis
## 5. ec_number_comparison.csv: Quality control for enzyme function preservation
## 6. filtering_summary_report.csv: Executive summary of filtering impact

## checking the integrity of the resulting filtered_df

# Make sure your dataframes were created successfully
head(filtered_df)
head(filtering_results)
nrow(filtered_df)
nrow(filtering_results)

