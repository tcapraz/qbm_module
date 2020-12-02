import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv(os.getcwd() + "/batchcorrected.csv", sep=",", index_col=0)
meta = pd.read_csv(os.getcwd() + "/meta_all.tsv", sep ="\t")

#split groups
sample_ids = meta["Sample_ID"]
y = meta["Group"]
y.index = sample_ids
data = data[sample_ids]


#hold out 20% of data for final evaluation
data,X_finaltest, y, y_finaltest = train_test_split(data.T,y, test_size=0.2, stratify = y)
data = data.T
crc = y[y=="CRC"]
ctr = y[y=="CTR"]
crcdata = data[crc.index]
ctrdata = data[ctr.index]


#Wilcoxon test for each species
pvals = []
for i in range(crcdata.shape[0]):
    pvals.append(ranksums(crcdata.iloc[i,:],ctrdata.iloc[i,])[1])
#multiple testing correction using benjamini hochberg
cor = multipletests(pvals, method="fdr_bh", alpha = 0.05)

#calculate log2FC
meanscrc = np.mean(np.exp(crcdata), axis =1)
meansctr = np.mean(np.exp(ctrdata), axis =1)
log2crc = np.log2(meanscrc)
log2ctr = np.log2(meansctr)
log2crc[log2crc==-np.inf] = 0
log2ctr[log2ctr==-np.inf] = 0

log2FC = log2crc - log2ctr

#plot volcano plot
sigidx = []
for i in range(log2FC.shape[0]):
    if abs(log2FC.iloc[i]) >= 2 and cor[1][i] <= 0.05:
        sigidx.append(i)
        
sigFC = log2FC.iloc[sigidx]
sigpval = cor[1][sigidx]
threshold = -np.log10(0.05)

plt.scatter(log2FC, -np.log10(cor[1]))
plt.scatter(sigFC, -np.log10(sigpval))
plt.plot([-20,20], [threshold,threshold], "--" , c = "orange")
plt.plot([-2,-2], [0,9], "--" , c = "orange")
plt.plot([2,2], [0,9], "--" , c = "orange")
plt.xlabel("log2 fold change CRC/CTR")
plt.ylabel("-log10 p-value")
plt.savefig(os.getcwd() + "/markergenus.png")
plt.show()
plt.close()

with open(os.getcwd()+ "/markergenus.txt","w") as file:
    for i in sigFC.index:
        file.write(str(i)+"\n")

#subset data with markers for ML
X = data.loc[sigFC.index].T
X_finaltest = X_finaltest.T.loc[sigFC.index].T
names = ["logr", "rf", "svm", "knn", "nb"]       
param_grids =[{"tol":np.logspace(-6,-1,num=6, dtype=float), "C": np.logspace(1,3, num= 3) }, 
              {"n_estimators" : np.linspace(50,1000, num =20, dtype=int)}, 
              {"tol":np.logspace(-6,-1,num=6, dtype=float), "C": np.logspace(1,6, num= 6)},
              {"n_neighbors" : np.linspace(5,50,num=10, dtype=int)},
              {"var_smoothing" : [1e-9]}]
res = {}



best_params = []
kf = StratifiedKFold(shuffle=True)
models = [LogisticRegression(), RandomForestClassifier(), SVC(),  KNeighborsClassifier(), GaussianNB()]          
for clf,name,params in zip(models,names,param_grids):
    res[name] = []
    gridCV = GridSearchCV(clf, params, cv= kf, n_jobs=-1)
    gridCV.fit(X, y)
    res[name].append(gridCV.best_score_)
    best_params.append(gridCV.best_params_)


#plot accuracies over the runs
accs = []
for i in names:
    accs.append(res[i][0])

plt.bar(names, accs, width=0.5)
plt.ylabel("Cross validated accuracy")
plt.savefig(os.getcwd() + "/cv_acc.png")
plt.show()
plt.close()

finalaccs = []
finalprobas = []
models = [LogisticRegression(**best_params[0]), RandomForestClassifier(**best_params[1]), 
          SVC(**best_params[2]),  KNeighborsClassifier(**best_params[3]), GaussianNB(**best_params[4])]          

for clf,name in zip(models,names):
    clf.fit(X,y)
    y_finalpred = clf.predict(X_finaltest)
    finalaccs.append(accuracy_score(y_finalpred, y_finaltest))
    if name =="svm":
        finalprobas.append(clf.decision_function(X_finaltest))
    else:
        finalprobas.append(clf.predict_proba(X_finaltest))

plt.bar(names, finalaccs, width=0.5)
plt.ylabel("Final accuracy")
plt.savefig(os.getcwd() + "/final_acc.png")
plt.show()
plt.close()

y_finaltest[y_finaltest=="CRC"] = 0 
y_finaltest[y_finaltest=="CTR"] = 1 

outdfs = []
for i in finalprobas:
    outdfs.append(pd.concat([pd.DataFrame(i, index =y_finaltest.index),y_finaltest], axis=1))
for i,name in zip(outdfs,names):
    i.to_csv(os.getcwd() +"/" +name +".csv")