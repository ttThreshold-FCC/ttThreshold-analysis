import uproot
import pandas as pd
import numpy as np
import os,sys
import re 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
sys.path.insert(0, "/afs/cern.ch/work/a/anmehta/public/FCC_ver2/FCCAnalyses/ttThreshold-analysis/") #os.getcwd()+'../')
from treemaker_WbWb_reco import all_branches
#from sklearn.feature_selection import chi2
#from scipy.stats import chisquare

def if3(cond, iftrue, iffalse):
    return iftrue if cond else iffalse



ecm = '345'
nJ='2' 
pf="sig_vs_wwz_nombb"
max_entries = 1E05
hadronic = False
channel =  if3(nJ == '0','lep',if3(nJ=='1', 'semihad','had')) 
base_dir=f"/eos/cms/store/cmst3/group/top//FCC_tt_threshold/output_condor_20241114_2154/WbWb/{channel}"
sig=f"wzp6_ee_WbWb_ecm{ecm}"
bkg=f"wzp6_ee_WWZ_Zbb_ecm{ecm}"

branches_toTrain=['jet1_R5_p','jet2_R5_p','jet1_R5_theta','jet2_R5_theta','nbjets_R5_eff_p9','mbbar']
if channel == "semihad":
    branches_toTrain+=['missing_p','lep_p','lep_theta']

#selection="lambda arrays: arrays [nbjets_R5_eff_p9] > 1"
selection="nbjets_R5_eff_p9 > 1"

infile_s =  uproot.concatenate([os.path.join(base_dir,sig,f)+':events' for f in os.listdir(os.path.join(base_dir,sig)) if f.endswith(".root")],branches_toTrain,cut=selection,library="np")
infile_b =  uproot.concatenate([os.path.join(base_dir,bkg,f)+':events' for f in os.listdir(os.path.join(base_dir,bkg)) if f.endswith(".root")],branches_toTrain,cut=selection,library="np")

print("for \t",channel,"\t channel used branches are\t",branches_toTrain)

#infile_s=infile_s.pop('nbjets_R5_eff_p9')
#infile_b=infile_b.pop('nbjets_R5_eff_p9')
pd_s = pd.DataFrame(infile_s).drop(columns=['nbjets_R5_eff_p9'])
pd_b = pd.DataFrame(infile_b).drop(columns=['nbjets_R5_eff_p9'])


print(type(pd_s))
pd_s['target'] = 1.
pd_b['target'] = 0.

pd_all = pd.concat([pd_s, pd_b])



X_train, X_test, y_train, y_test = train_test_split(pd_all.drop(columns=['target']), pd_all['target'], test_size=0.5) #train_test_split
print('starting to train with features',X_train.columns,len(X_train.columns))
bst = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, verbosity=0, objective='binary:logistic')
print('created model instance ')
bst.fit(X_train, y_train)
print('fitted the model')
preds = bst.predict_proba(X_test)
print('predictions')
pd_preds = pd.DataFrame()
pd_preds['target'] = y_test
pd_preds['preds'] = preds[:,1]


train_preds = bst.predict_proba(X_train)
pd_train_preds = pd.DataFrame()
pd_train_preds['target']  = y_train
pd_train_preds['preds'] = train_preds[:,1]


import matplotlib.pyplot as plt

# Plot signal and background predictions

bins=np.linspace(0,1,51)
n,bins,_=plt.hist(pd_preds[pd_preds['target'] == 1]['preds'], bins=bins, label='Signal-test', histtype='step',color='red', log=False)
mid = 0.5*(bins[1:] + bins[:-1])
plt.errorbar(mid, n, yerr=np.sqrt(n), fmt='none',ecolor='red')

n1,bins1,_=plt.hist(pd_preds[pd_preds['target'] == 0]['preds'], bins=bins, label='Background-test', histtype='step',color='blue', log=False)
mid1 = 0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(mid1, n1, yerr=np.sqrt(n1), fmt='none',ecolor='blue')
#ax.errorbar(bin_mids, n_SM, yerr = calculate_stat_bin_ucts(Y_pred[Y_test==0], w_SM, bins), fmt='none', ecolor='black', capsize=0)


n2,bins2,_=plt.hist(pd_train_preds[pd_train_preds['target'] == 1]['preds'], bins=bins, label='Signal-train', histtype='step',color='green', log=False)
mid2 = 0.5*(bins2[1:] + bins2[:-1])
plt.errorbar(mid2, n2, yerr=np.sqrt(n2), fmt='none',ecolor='green')

n3,bins3,_=plt.hist(pd_train_preds[pd_train_preds['target'] == 0]['preds'], bins=bins, label='Background-train',  histtype='step',color='orange', log=False)
mid3 = 0.5*(bins3[1:] + bins3[:-1])
plt.errorbar(mid3, n3, yerr=np.sqrt(n3), fmt='none',ecolor='orange')


plt.xlabel('Prediction')
plt.ylabel('Count')
plt.legend()
print('Saving model')
outdir = '/eos/cms/store/cmst3/group/top//FCC_tt_threshold/BDT_model/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

bst.save_model('{}/model_{}_{}{}.json'.format(outdir,channel,ecm,pf))
plt.savefig('{}/preds_{}_{}{}.png'.format(outdir,channel,ecm,pf))
plt.savefig('{}/preds_{}_{}{}.pdf'.format(outdir,channel,ecm,pf))
plt.close()