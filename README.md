This script reads in metagenomics data, normalizes it to relative counts 
and performs log transformation.
To find marker species for a given disease it calculates the wilcoxon rank sums test for each
between the two groups and corrects for multiple testing.
Markerspecies are defined as having a corrected p-value <= 0.05 and a |log2FC| >= 2.
Markerspecies are written to markerspecies.txt and a volcano plot is plotted for visualization.
