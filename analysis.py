import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
"""
This script reads in metagenomics data, normalizes it to relative counts 
and performs log transformation.
To find marker species for a given disease it calculates the wilcoxon rank sums test for each
between the two groups and corrects for multiple testing.
Markerspecies are defined as having a corrected p-value <= 0.05 and a |log2FC| >= 2.
Markerspecies are written to markerspecies.txt and a volcano plot is plotted for visualization.
"""

#read in data
feat = pd.read_csv(os.getcwd() + "/data/feat_all.tsv", sep="\t")
meta = pd.read_csv(os.getcwd() + "/data/meta_all.tsv", sep ="\t")

#Normalization
normfeat = feat/feat.sum(axis=0)

#log transformation
lognormfeat = np.log(normfeat[normfeat!=0]).fillna(0)

#split groups
sample_ids = meta["Sample_ID"]
y = meta["Group"]
y.index = sample_ids
crc = y[y=="CRC"]
ctr = y[y=="CTR"]
data = lognormfeat[sample_ids]
crcdata = data[crc.index]
ctrdata = data[ctr.index]

#Wilcoxon test for each species
pvals = []
for i in range(crcdata.shape[0]):
    pvals.append(ranksums(crcdata.iloc[i,:],ctrdata.iloc[i,])[1])
#multiple testing correction using benjamini hochberg
cor = multipletests(pvals, method="fdr_bh", alpha = 0.05)

#calculate log2FC
meanscrc = np.mean(normfeat[sample_ids][crc.index], axis =1)
meansctr = np.mean(normfeat[sample_ids][ctr.index], axis =1)
log2crc = np.log2(meanscrc)
log2ctr = np.log2(meansctr)
log2crc[log2crc==-np.inf] = 0
log2ctr[log2ctr==-np.inf] = 0

log2FC = log2crc - log2ctr

#plot volcano plot
sigFC = log2FC.iloc[np.where(cor[1] <= 0.05)[0]]
sigpval = cor[1][np.where(cor[1] <= 0.05)[0]]
threshold = -np.log10(0.05)

plt.scatter(log2FC, -np.log10(cor[1]))
plt.scatter(sigFC, -np.log10(sigpval))
plt.plot([-20,20], [threshold,threshold], "--" , c = "orange")
plt.plot([-2,-2], [0,4.5], "--" , c = "orange")
plt.plot([2,2], [0,4.5], "--" , c = "orange")
plt.xlabel("log2 fold change")
plt.ylabel("-log10 p-value")
plt.savefig(os.getcwd() + "/markerspecies.png")
plt.show()
plt.close()

#write markerspecies to txt file
with open(os.getcwd()+ "/markerspecies.txt","w") as file:
    for i in sigFC.index:
        file.write(str(i)+"\n")


