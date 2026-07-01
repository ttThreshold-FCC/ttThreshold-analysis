#!/usr/bin/env python3
# Dump reco jets + gen pairing + C++ kinfit branches from a 4q step2 root file
# to .npz, for the JAX kinfit prototype (head-to-head on identical events).
# Run with the FCC/default python (uproot). Usage: dump_jets_npz.py <in.root> <out.npz>
import sys, uproot, numpy as np
t = uproot.open(sys.argv[1])["events"]
br = ["reco_jet1_p","reco_jet1_theta","reco_jet1_phi",
      "reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
      "reco_jet3_p","reco_jet3_theta","reco_jet3_phi",
      "reco_jet4_p","reco_jet4_theta","reco_jet4_phi"]
opt = ["gen_pairing_true","kinfit4q_valid","kinfit4q_valid_loose",
       "kinfit4q_pairing_correct","kinfit4q_chi2","kinfit4q_pairing","kinfit4q_mW"]
have = set(t.keys())
keep = [b for b in br+opt if b in have]
a = t.arrays(keep, library="np")
np.savez(sys.argv[2], **a)
print(f"dumped {len(a[keep[0]])} events, branches: {keep}")
