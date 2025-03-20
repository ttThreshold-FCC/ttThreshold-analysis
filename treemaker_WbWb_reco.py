import os, copy
import urllib

# list of processes
all_processes = {
#    "p8_ee_ZZ_ecm365":{ "fraction": 1,},
#    "p8_ee_ZZ_ecm345":{ "fraction": 1,},
#    "p8_ee_ZZ_ecm340":{ "fraction": 1,},
#    "wzp6_ee_qq_PSdown_ecm340" :{ "fraction": 1,},
#    "wzp6_ee_qq_PSup_ecm345":{ "fraction": 1,},
#    "wzp6_ee_qq_PSdown_ecm345":{ "fraction": 1,},
#    "wzp6_ee_qq_PSup_ecm365":{ "fraction": 1,},
#    "wzp6_ee_qq_PSdown_ecm365":{ "fraction": 1,},
#    "wzp6_ee_qq_PSup_ecm340":{ "fraction": 1,},
#    
#    "wzp6_ee_WWZ_Zbb_ecm340": {
#        "fraction": 1,
#     },
#    "wzp6_ee_WWZ_Zbb_ecm345": {
#        "fraction": 1,
#     },
#    "wzp6_ee_WWZ_Zbb_ecm365": {
#        "fraction": 1,
#     },
#
#     "wzp6_ee_WbWb_ecm340": {
#         "fraction": 1,
#     },
#   "wzp6_ee_WbWb_ecm345": {
#       "fraction": 1,
#   },
#    "wzp6_ee_WbWb_ecm350": {
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_ecm355": {
#        "fraction": 1,
#    },
#
#    "wzp6_ee_WbWb_ecm365": {
#        "fraction": 1,
#    },
    "p8_ee_WW_ecm345": {
        "fraction": 1,
    },
#    "p8_ee_WW_ecm365": {
#         "fraction": 1,
#    },    
#    "p8_ee_WW_ecm350": {
#       "fraction": 1,
#    },
#    "p8_ee_WW_ecm340": {
#       "fraction": 1,
#    },
#     "p8_ee_WW_ecm355": {
#        "fraction": 1,
#     },
#
#    "wzp6_ee_WbWb_PSup_ecm345":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_PSdown_ecm340":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_PSdown_ecm345":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_PSup_ecm365":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_PSdown_ecm365":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_PSup_ecm340":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_mtop171p5_ecm345":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_mtop171p5_ecm340":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_mtop173p5_ecm345":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_mtop171p5_ecm365":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_mtop173p5_ecm340":{
#        "fraction": 1,
#    },
#    "wzp6_ee_WbWb_mtop173p5_ecm365":{
#        "fraction": 1,
#    },
#    "wzp6_ee_qq_ecm365":{
#       "fraction": 1,
#    },    
#    "wzp6_ee_qq_ecm345": {
#       "fraction": 1,
#    },
#    "wzp6_ee_qq_ecm340": {
#        "fraction": 1,
#    },
#
#
###FOR foll WW background samples with PS variations xsec values are wrong in the database!! 
#   "p8_ee_WW_PSdown_ecm340":{ "fraction": 1,},
#   "p8_ee_WW_PSup_ecm340":{ "fraction": 1,},
#   "p8_ee_WW_PSdown_ecm345":{ "fraction": 1,},
#   "p8_ee_WW_PSup_ecm345":{ "fraction": 1,},
#   "p8_ee_WW_PSdown_ecm365":{ "fraction": 1,},
#   "p8_ee_WW_PSup_ecm365":{ "fraction": 1,},
    
###### following samples we don't use
##am     "wzp6_ee_WbWb_semihad_ecm345": {
##am         "fraction": 1,
##am     },
##am     "wzp6_ee_WbWb_had_ecm345": {
##am         "fraction": 1,
##am     },
##am    "wzp6_ee_WbWb_semihad_ecm350": {
##am       "fraction": 1,
##am    },
##am    "wzp6_ee_WbWb_had_ecm350": {
##am       "fraction": 1,
##am    },
##am    "wzp6_ee_WbWb_semihad_ecm355": {
##am        "fraction": 1,
##am    },
##am    "wzp6_ee_WbWb_had_ecm355": {
##am        "fraction": 1,
##am    },
##am    "wzp6_ee_WbWb_semihad_ecm340": {
##am        "fraction": 1,
##am    },
##am    "wzp6_ee_WbWb_had_ecm340": {
##am        "fraction": 1,
##am    },
##am    "wzp6_ee_WbWb_lep_ecm340": {
##am     "fraction": 1,
##am     },
##am    "wzp6_ee_WbWb_lep_ecm345": {
##am             "fraction": 1,
##am     },
##am    "wzp6_ee_WbWb_lep_ecm350": {
##am     "fraction": 1,
##am     },
##am    "wzp6_ee_WbWb_lep_ecm355": {
##am             "fraction": 1,
##am     },
##am    "wzp6_ee_WbWb_lep_ecm365": {
##am         "fraction": 1,
##am     },
##am    "wzp6_ee_WbWb_semihad_ecm365": {
##am        "fraction": 1,
##am     },
##am    "wzp6_ee_WbWb_had_ecm365": {
##am         "fraction": 1,
##am    },

##     "wzp6_ee_WbWb_had_ecm345": {
##         "fraction": 1,
##     },
##    "wzp6_ee_WbWb_semihad_ecm350": {
##       "fraction": 1,
##    },
##    "wzp6_ee_WbWb_had_ecm350": {
##       "fraction": 1,
##    },
##    "wzp6_ee_WbWb_semihad_ecm355": {
##        "fraction": 1,
##    },
##    "wzp6_ee_WbWb_had_ecm355": {
##        "fraction": 1,
##    },
##    "wzp6_ee_WbWb_semihad_ecm340": {
##        "fraction": 1,
##    },
##    "wzp6_ee_WbWb_had_ecm340": {
##        "fraction": 1,
##    },
##
##    "wzp6_ee_WbWb_lep_ecm340": {
##     "fraction": 1,
##     },
##    "wzp6_ee_WbWb_lep_ecm345": {
##             "fraction": 1,
##     },
##    "wzp6_ee_WbWb_lep_ecm350": {
##     "fraction": 1,
##     },
##    "wzp6_ee_WbWb_lep_ecm355": {
##             "fraction": 1,
##     },
    #"wzp6_ee_WbWb_lep_ecm365": {
    #     "fraction": 1,
    # },
    #"wzp6_ee_WbWb_semihad_ecm365": {
    #    "fraction": 1,
    # },
    #"wzp6_ee_WbWb_had_ecm365": {
    #     "fraction": 1,
    #},
    #"wzp6_ee_WbWb_semihad_mtop173p5_ecm365": {
    #     "fraction": 1,
    #},
    #
    #"wzp6_ee_WbWb_semihad_mtop171p5_ecm365": {
    #     "fraction": 1,
    #},


    
}

available_ecm = ['340','345', '350', '355','365']

hadronic  = False
#semihad  = False
#lep      = False
ecm       = 345
print(ecm)

saveExclJets = False

if not str(ecm) in available_ecm:
    raise ValueError("ecm value not in available_ecm")

channel = "CHANNELNAMEHERE"

if  channel not in ["lep","semihad","had"]:
    print("using defa channel settings")
    channel="semihad"
print(channel)    

processList={key: value for key, value in all_processes.items() if str(ecm) in available_ecm and str(ecm) in key } # (True if str('p8_ee_WW_ecm'+ecm) in key else str('wzp6_ee_WbWb_ecm'+ecm) in key)}  

print(processList)
# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
prodTag     = "FCCee/winter2023/IDEA/"

#Optional: output directory, default is local running directoryp
outputDir   = "outputs/treemaker/WbWb/{}".format(channel)


# additional/costom C++ functions, defined in header files (optional)
includePaths = ["examples/functions.h"]

## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc" #"fccee_flavtagging_edm4hep_wc_v1"

## model files locally stored on /eos
eos_dir ="/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"
model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"    # "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

## get local file, else download from url
def get_file_path(url, filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)

def get_files(eos_dir, proc):
    files=[]
    basepath=os.path.join(eos_dir,proc)
    if os.path.exists(basepath):
        files =  [os.path.join(basepath,x) for x in os.listdir(basepath) if os.path.isfile(os.path.join(basepath, x)) ]
    return files

weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

#inputDir = eos_dir

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
    InclusiveJetClusteringHelper,
)

jetFlavourHelper = None
jetFlavourHelper_R5 = None
jetClusteringHelper = None
jetClusteringHelper_R5 = None

all_branches = [
    "nlep", "lep_p", 'lep_theta', 'lep_phi',
    "missing_p", "missing_p_theta", "missing_p_phi",
    "njets_R5",  "jet1_R5_p", "jet2_R5_p", "jet3_R5_p", "jet4_R5_p", "jet5_R5_p", "jet6_R5_p",
    "jet1_R5_theta",  "jet2_R5_theta",  "jet3_R5_theta",  "jet4_R5_theta", "jet5_R5_theta", "jet6_R5_theta",
    "jet1_R5_pflavor", "jet2_R5_pflavor", "jet3_R5_pflavor", "jet4_R5_pflavor", "jet5_R5_pflavor","jet6_R5_pflavor",
    "nbjets_R5_true", "ncjets_R5_true","nljets_R5_true","ngjets_R5_true",
    "nbjets_R5_eff_p9", "nbjets_R5_eff_p89","nbjets_R5_eff_p91",
    "jet1_R5_isG","jet2_R5_isG","jet3_R5_isG","jet4_R5_isG","jet5_R5_isG","jet6_R5_isG",                
    "jet1_R5_isU","jet2_R5_isU","jet3_R5_isU","jet4_R5_isU","jet5_R5_isU","jet6_R5_isU",                
    "jet1_R5_isB","jet2_R5_isB","jet3_R5_isB","jet4_R5_isB","jet5_R5_isB","jet6_R5_isB",                
    "jet1_R5_isS","jet2_R5_isS","jet3_R5_isS","jet4_R5_isS","jet5_R5_isS","jet6_R5_isS",                
    "jet1_R5_isC","jet2_R5_isC","jet3_R5_isC","jet4_R5_isC","jet5_R5_isC","jet6_R5_isC",                
    "jet1_R5_isD","jet2_R5_isD","jet3_R5_isD","jet4_R5_isD","jet5_R5_isD","jet6_R5_isD",                                
    "jet1_R5_isTAU","jet2_R5_isTAU","jet3_R5_isTAU","jet4_R5_isTAU","jet5_R5_isTAU","jet6_R5_isTAU","mbbar_p9","mbbar_p89","mbbar_p91", "bjet1_R5_true_p","ljet1_R5_true_p",
    ]

if saveExclJets: all_branches+=[ "njets", "jet1_p", "jet2_p", "jet3_p","jet4_p","jet5_p","jet6_p",
    "jet1_theta", "jet2_theta", "jet3_theta","jet4_theta","jet5_theta","jet6_theta",
    "jet1_phi", "jet2_phi", "jet3_phi","jet4_phi","jet5_phi","jet6_phi",
    "jet1_isTau", "jet2_isTau", "jet3_isTau","jet4_isTau","jet5_isTau","jet6_isTau",
    "nbjets_WPp5", "nbjets_WPp8",
    "nbjets_WPp85", "nbjets_WPp9",
    "d_12","d_23","d_34","d_45","d_56",
    "jet1_isB", "jet2_isB", "jet3_isB", "jet4_isB", "jet5_isB", "jet6_isB",
    "jet1_isG", "jet2_isG", "jet3_isG", "jet4_isG", "jet5_isG", "jet6_isG",
    "jet1_isQ", "jet2_isQ", "jet3_isQ", "jet4_isQ", "jet5_isQ", "jet6_isQ",
    "jet1_isS", "jet2_isS", "jet3_isS", "jet4_isS", "jet5_isS", "jet6_isS",
    "jet1_isC", "jet2_isC", "jet3_isC", "jet4_isC", "jet5_isC", "jet6_isC",
    "nbjets_R5_WPp5", "nbjets_R5_WPp8", "nbjets_R5_WPp85", "nbjets_R5_WPp9"]
#print('saving these branches',all_branches)
# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:

    # __________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):

        # __________________________________________________________
        # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2

        # define some aliases to be used later on
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Electron0","Electron#0.index")

        # get all the leptons from the collection
        df = df.Define(
            "muons_all",
            "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)",
        )
        df = df.Define(
            "electrons_all",
            "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)",
        )

        # select leptons with momentum > 12 GeV
        df = df.Define(
            "muons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p(12)(muons_all)",
        )

        df = df.Define(
            "electrons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p(12)(electrons_all)",
        )

        # compute the muon isolation and store muons with an isolation cut of 0df = df.25 in a separate column muons_sel_iso
        df = df.Define(
            "muons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons_sel, ReconstructedParticles)",
        )
        # compute the electron isolation and store electrons with an isolation cut of 0df = df.25 in a separate column electrons_sel_iso
        df = df.Define(
            "electrons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons_sel, ReconstructedParticles)",
        )

        df = df.Define(
            "muons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.7)(muons_sel, muons_iso)",
        )

        df = df.Define(
            "electrons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.7)(electrons_sel, electrons_iso)",
        )

        if channel == "had":
            #hadronic=True
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 0")
        elif  channel == "semihad":
            #semihad=True
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 1")
        else:
            #lep=True
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 2")

        if not (channel == "had"):
            df = df.Define(
                "muons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons_sel_iso)"
            )


            df = df.Define(
                "electrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons_sel_iso)"
            )

            df = df.Define(
                "muons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(muons_sel_iso)",
            )
            df = df.Define(
                "muons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(muons_sel_iso)",
            )
            df = df.Define(
                "muons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(muons_sel_iso)",
            )
            df = df.Define(
                "muons_n", "FCCAnalyses::ReconstructedParticle::get_n(muons_sel_iso)",
            )

            df = df.Define(
                "electrons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(electrons_sel_iso)",
            )
            df = df.Define(
                "electrons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(electrons_sel_iso)",
                )
            df = df.Define(
                "electrons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(electrons_sel_iso)",
                )
            df = df.Define(
                "electrons_n", "FCCAnalyses::ReconstructedParticle::get_n(electrons_sel_iso)",
            )

        
        ## here cluster jets in the events but first remove muons from the list of
        ## reconstructed particles
        
        ## create a new collection of reconstructed particles removing muons with p>12
        df = df.Define(
            "ReconstructedParticlesNoMuons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles,muons_sel_iso)",
        )
        df = df.Define(
            "ReconstructedParticlesNoMuNoEl",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoMuons,electrons_sel_iso)",
        )


        ## perform exclusive jet clustering
        global jetClusteringHelper
        global jetFlavourHelper
        global jetFlavourHelper_R5
        global jetClusteringHelper_R5
        
        ## define jet and run clustering parameters
        ## name of collections in EDM root files
        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "EFlowTrack_1",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "EFlowTrack_2",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }

        nJets = 4 if  channel == "semihad" else 6

        collections_noleps = copy.deepcopy(collections)
        collections_noleps["PFParticles"] = "ReconstructedParticlesNoMuNoEl"
        if saveExclJets:
            jetClusteringHelper = ExclusiveJetClusteringHelper(
                collections_noleps["PFParticles"], nJets
            )
        
        jetClusteringHelper_R5  = InclusiveJetClusteringHelper(
            collections_noleps["PFParticles"], 0.5, 10, "R5"
        )
        if  saveExclJets:
            df = jetClusteringHelper.define(df)
        df = jetClusteringHelper_R5.define(df)
        ## define jet flavour tagging parameters
        if saveExclJets:
            jetFlavourHelper = JetFlavourHelper(
                collections_noleps,
                jetClusteringHelper.jets,
                jetClusteringHelper.constituents,
            )
        jetFlavourHelper_R5 = JetFlavourHelper(
            collections_noleps,
            jetClusteringHelper_R5.jets,
            jetClusteringHelper_R5.constituents,
            "R5",
        )
        if saveExclJets:    df = jetFlavourHelper.define(df)
        
        df = jetFlavourHelper_R5.define(df)
        ## tagger inference
        if  saveExclJets: df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)
        df = jetFlavourHelper_R5.inference(weaver_preproc, weaver_model,df)

        df = df.Define(
            "lep_p", "muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_p(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_p(electrons_sel_iso)[0] : -999) "
        )
        df = df.Define(
            'lep_theta', 'muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_theta(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_theta(electrons_sel_iso)[0] : -999) '
        )
        df = df.Define(
            'lep_phi', 'muons_sel_iso.size() >0 ? FCCAnalyses::ReconstructedParticle::get_phi(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_phi(electrons_sel_iso)[0] : -999) '
        )
        df = df.Define(
            "nlep",
            "electrons_sel_iso.size()+muons_sel_iso.size()")
        


        df = df.Define(
            "missing_p",
            "FCCAnalyses::ReconstructedParticle::get_p(MissingET)[0]",
        )

        
        df = df.Define(
            'missing_p_theta', 'ReconstructedParticle::get_theta(MissingET)[0]',
        )

        df = df.Define(
            'missing_p_phi', 'ReconstructedParticle::get_phi(MissingET)[0]',
        )

        if    saveExclJets:
            df = df.Define(
                "jets_p4",
                "JetConstituentsUtils::compute_tlv_jets({})".format(
                    jetClusteringHelper.jets
                ),
            )
        
        df = df.Define(
            "jets_R5_p4",
            "JetConstituentsUtils::compute_tlv_jets({})".format(
                jetClusteringHelper_R5.jets
            ),
        )

        df = df.Define("jets_R5_p",           "JetClusteringUtils::get_p({})".format(jetClusteringHelper_R5.jets))
        df = df.Define("jets_R5_theta",       "JetClusteringUtils::get_theta({})".format(jetClusteringHelper_R5.jets))
        df = df.Define("jet1_R5_p","jets_R5_p[0]")
        df = df.Define("jet2_R5_p","jets_R5_p[1]")
        df = df.Define("jet3_R5_p","jets_R5_p.size()>2 ? jets_R5_p[2] : -999")
        df = df.Define("jet4_R5_p","jets_R5_p.size()>3 ? jets_R5_p[3] : -999")
        df = df.Define("jet5_R5_p","jets_R5_p.size()>4 ? jets_R5_p[4] : -999")
        df = df.Define("jet6_R5_p","jets_R5_p.size()>5 ? jets_R5_p[5] : -999")

        df = df.Define("jet1_R5_theta","jets_R5_theta[0]")
        df = df.Define("jet2_R5_theta","jets_R5_theta[1]")
        df = df.Define("jet3_R5_theta","jets_R5_theta.size()>2 ? jets_R5_theta[2] : -999")
        df = df.Define("jet4_R5_theta","jets_R5_theta.size()>3 ? jets_R5_theta[3] : -999")
        df = df.Define("jet5_R5_theta","jets_R5_theta.size()>4 ? jets_R5_theta[4] : -999")
        df = df.Define("jet6_R5_theta","jets_R5_theta.size()>5 ? jets_R5_theta[5] : -999")
        
        df = df.Define("jets_R5_pflavor", "JetTaggingUtils::get_flavour({}, Particle)".format(jetClusteringHelper_R5.jets) )
        df = df.Define("jet1_R5_pflavor","jets_R5_pflavor[0]")
        df = df.Define("jet2_R5_pflavor","jets_R5_pflavor[1]")
        df = df.Define("jet3_R5_pflavor","jets_R5_p.size()>2 ? jets_R5_pflavor[2] : -999")
        df = df.Define("jet4_R5_pflavor","jets_R5_p.size()>3 ? jets_R5_pflavor[3] : -999")
        df = df.Define("jet5_R5_pflavor","jets_R5_p.size()>4 ? jets_R5_pflavor[4] : -999")
        df = df.Define("jet6_R5_pflavor","jets_R5_p.size()>5 ? jets_R5_pflavor[5] : -999")
        df = df.Define("njets_R5",       "return int(jets_R5_pflavor.size())")


        

        
        
        df = df.Define("jets_R5_btag_true", "JetTaggingUtils::get_btag({}, 1.0)".format('jets_R5_pflavor'))
        df = df.Define("jets_R5_ctag_true", "JetTaggingUtils::get_ctag({}, 1.0)".format('jets_R5_pflavor'))
        df = df.Define("jets_R5_ltag_true", "JetTaggingUtils::get_ltag({}, 1.0)".format('jets_R5_pflavor'))
        df = df.Define("jets_R5_gtag_true", "JetTaggingUtils::get_gtag({}, 1.0)".format('jets_R5_pflavor'))
        

        df = df.Define("jets_R5_btag_eff_p9", "JetTaggingUtils::get_btag({}, .9)".format('jets_R5_pflavor'))
        df = df.Define("jets_R5_btag_eff_p89", "JetTaggingUtils::get_btag({}, .89)".format('jets_R5_pflavor'))
        df = df.Define("jets_R5_btag_eff_p91", "JetTaggingUtils::get_btag({}, .91)".format('jets_R5_pflavor'))

        df = df.Define("jets_R5_btagged_true", "JetTaggingUtils::sel_tag(true)(jets_R5_btag_true,{})".format(jetClusteringHelper_R5.jets))
        df = df.Define("jets_R5_ctagged_true", "JetTaggingUtils::sel_tag(true)(jets_R5_ctag_true,{})".format(jetClusteringHelper_R5.jets))
        df = df.Define("jets_R5_ltagged_true", "JetTaggingUtils::sel_tag(true)(jets_R5_ltag_true,{})".format(jetClusteringHelper_R5.jets))
        df = df.Define("jets_R5_gtagged_true", "JetTaggingUtils::sel_tag(true)(jets_R5_gtag_true,{})".format(jetClusteringHelper_R5.jets))

        df = df.Define("jets_R5_btagged_eff_p9", "JetTaggingUtils::sel_tag(true)(jets_R5_btag_eff_p9,{})".format(jetClusteringHelper_R5.jets))
        df = df.Define("jets_R5_btagged_eff_p89", "JetTaggingUtils::sel_tag(true)(jets_R5_btag_eff_p89,{})".format(jetClusteringHelper_R5.jets))
        df = df.Define("jets_R5_btagged_eff_p91", "JetTaggingUtils::sel_tag(true)(jets_R5_btag_eff_p91,{})".format(jetClusteringHelper_R5.jets))

        df = df.Define("nbjets_R5_true", "return int(jets_R5_btagged_true.size())")
        df = df.Define("ncjets_R5_true", "return int(jets_R5_ctag_true.size())")
        df = df.Define("nljets_R5_true", "return int(jets_R5_ltag_true.size())")
        df = df.Define("ngjets_R5_true", "return int(jets_R5_gtag_true.size())")


        df = df.Define("bjets_R5_true",  "JetConstituentsUtils::compute_tlv_jets({})". format('jets_R5_btagged_true'))
        df = df.Define("ljets_R5_true",  "JetConstituentsUtils::compute_tlv_jets({})". format('jets_R5_ltagged_true'))
        df = df.Define("bjet1_R5_true_p","nbjets_R5_true > 1 ? bjets_R5_true[0].P() : -999")
        df = df.Define("ljet1_R5_true_p","nljets_R5_true > 1 ? ljets_R5_true[0].P() : -999")

        

        
        df = df.Define("nbjets_R5_eff_p9", "return int(jets_R5_btagged_eff_p9.size())")
        df = df.Define("nbjets_R5_eff_p89", "return int(jets_R5_btagged_eff_p89.size())")
        df = df.Define("nbjets_R5_eff_p91", "return int(jets_R5_btagged_eff_p91.size())")

        df = df.Define("bjet_R5_eff_p9_p4",  "JetConstituentsUtils::compute_tlv_jets({})". format('jets_R5_btagged_eff_p9'))
        df = df.Define("bjet_R5_eff_p91_p4", "JetConstituentsUtils::compute_tlv_jets({})".format('jets_R5_btagged_eff_p91'))
        df = df.Define("bjet_R5_eff_p89_p4", "JetConstituentsUtils::compute_tlv_jets({})".format('jets_R5_btagged_eff_p89'))
        df = df.Define("mbbar_p9",  "nbjets_R5_eff_p9 >   1 ? JetConstituentsUtils::InvariantMass(bjet_R5_eff_p9_p4[0],  bjet_R5_eff_p9_p4[1])  : -999")
        df = df.Define("mbbar_p91", "nbjets_R5_eff_p91 >  1 ? JetConstituentsUtils::InvariantMass(bjet_R5_eff_p91_p4[0], bjet_R5_eff_p91_p4[1]) : -999")
        df = df.Define("mbbar_p89", "nbjets_R5_eff_p89 >  1 ? JetConstituentsUtils::InvariantMass(bjet_R5_eff_p89_p4[0], bjet_R5_eff_p89_p4[1]) : -999")

        

        df = df.Define("jet1_R5_isG", "recojet_isG_R5[0]")
        df = df.Define("jet2_R5_isG", "recojet_isG_R5[1]")
        df = df.Define("jet3_R5_isG", "jets_R5_p.size()>2 ? recojet_isG_R5[2] : -999")
        df = df.Define("jet4_R5_isG", "jets_R5_p.size()>3 ? recojet_isG_R5[3] : -999")
        df = df.Define("jet5_R5_isG", "jets_R5_p.size()>4 ? recojet_isG_R5[4] : -999")
        df = df.Define("jet6_R5_isG", "jets_R5_p.size()>5 ? recojet_isG_R5[5] : -999")

        df = df.Define("jet1_R5_isU", "recojet_isU_R5[0]")
        df = df.Define("jet2_R5_isU", "recojet_isU_R5[1]")
        df = df.Define("jet3_R5_isU", "jets_R5_p.size()>2 ? recojet_isU_R5[2] : -999")
        df = df.Define("jet4_R5_isU", "jets_R5_p.size()>3 ? recojet_isU_R5[3] : -999")
        df = df.Define("jet5_R5_isU", "jets_R5_p.size()>4 ? recojet_isU_R5[4] : -999")
        df = df.Define("jet6_R5_isU", "jets_R5_p.size()>5 ? recojet_isU_R5[5] : -999")

        df = df.Define("jet1_R5_isB", "recojet_isB_R5[0]")
        df = df.Define("jet2_R5_isB", "recojet_isB_R5[1]")
        df = df.Define("jet3_R5_isB", "jets_R5_p.size()>2 ? recojet_isB_R5[2] : -999")
        df = df.Define("jet4_R5_isB", "jets_R5_p.size()>3 ? recojet_isB_R5[3] : -999")
        df = df.Define("jet5_R5_isB", "jets_R5_p.size()>4 ? recojet_isB_R5[4] : -999")
        df = df.Define("jet6_R5_isB", "jets_R5_p.size()>5 ? recojet_isB_R5[5] : -999")
        
        df = df.Define("jet1_R5_isS", "recojet_isS_R5[0]")
        df = df.Define("jet2_R5_isS", "recojet_isS_R5[1]")
        df = df.Define("jet3_R5_isS", "jets_R5_p.size()>2 ? recojet_isS_R5[2] : -999")
        df = df.Define("jet4_R5_isS", "jets_R5_p.size()>3 ? recojet_isS_R5[3] : -999")
        df = df.Define("jet5_R5_isS", "jets_R5_p.size()>4 ? recojet_isS_R5[4] : -999")
        df = df.Define("jet6_R5_isS", "jets_R5_p.size()>5 ? recojet_isS_R5[5] : -999")

        df = df.Define("jet1_R5_isC", "recojet_isC_R5[0]")
        df = df.Define("jet2_R5_isC", "recojet_isC_R5[1]")
        df = df.Define("jet3_R5_isC", "jets_R5_p.size()>2 ? recojet_isC_R5[2] : -999")
        df = df.Define("jet4_R5_isC", "jets_R5_p.size()>3 ? recojet_isC_R5[3] : -999")
        df = df.Define("jet5_R5_isC", "jets_R5_p.size()>4 ? recojet_isC_R5[4] : -999")
        df = df.Define("jet6_R5_isC", "jets_R5_p.size()>5 ? recojet_isC_R5[5] : -999")

        df = df.Define("jet1_R5_isD", "recojet_isD_R5[0]")
        df = df.Define("jet2_R5_isD", "recojet_isD_R5[1]")
        df = df.Define("jet3_R5_isD", "jets_R5_p.size()>2 ? recojet_isD_R5[2] : -999")
        df = df.Define("jet4_R5_isD", "jets_R5_p.size()>3 ? recojet_isD_R5[3] : -999")
        df = df.Define("jet5_R5_isD", "jets_R5_p.size()>4 ? recojet_isD_R5[4] : -999")
        df = df.Define("jet6_R5_isD", "jets_R5_p.size()>5 ? recojet_isD_R5[5] : -999")

        df = df.Define("jet1_R5_isTAU", "recojet_isTAU_R5[0]")
        df = df.Define("jet2_R5_isTAU", "recojet_isTAU_R5[1]")
        df = df.Define("jet3_R5_isTAU", "jets_R5_p.size()>2 ? recojet_isTAU_R5[2] : -999")
        df = df.Define("jet4_R5_isTAU", "jets_R5_p.size()>3 ? recojet_isTAU_R5[3] : -999")
        df = df.Define("jet5_R5_isTAU", "jets_R5_p.size()>4 ? recojet_isTAU_R5[4] : -999")
        df = df.Define("jet6_R5_isTAU", "jets_R5_p.size()>5 ? recojet_isTAU_R5[5] : -999")

        df = df.Define("jets_R5_isB","recojet_isB_R5")
        
        df = df.Define("bjets_R5_WPp5","ZHfunctions::sel_btag(0.5)(jets_R5_isB)")
        df = df.Define("bjets_R5_WPp8","ZHfunctions::sel_btag(0.8)(jets_R5_isB)")
        df = df.Define("bjets_R5_WPp85","ZHfunctions::sel_btag(0.85)(jets_R5_isB)")
        df = df.Define("bjets_R5_WPp9","ZHfunctions::sel_btag(0.9)(jets_R5_isB)")

        df = df.Define("nbjets_R5_WPp5","bjets_R5_WPp5.size()")
        df = df.Define("nbjets_R5_WPp8","bjets_R5_WPp8.size()")
        df = df.Define("nbjets_R5_WPp85","bjets_R5_WPp85.size()")
        df = df.Define("nbjets_R5_WPp9","bjets_R5_WPp9.size()")

        
        #['recojet_isG_R5', 'recojet_isU_R5', 'recojet_isS_R5', 'recojet_isC_R5', 'recojet_isB_R5', 'recojet_isTAU_R5', 'recojet_isD_R5', 'jet_nmu_R5', 'jet_nel_R5', 'jet_nchad_R5', 'jet_ngamma_R5', 'jet_nnhad_R5']
        if  saveExclJets:
            df = df.Define("jets_isB",   "JetFlavourUtils::get_weight(MVAVec_, 4)")
            df = df.Define("bjets_WPp5", "ZHfunctions::sel_btag(0.5)(jets_isB)")
            df = df.Define("bjets_WPp8", "ZHfunctions::sel_btag(0.8)(jets_isB)")
            df = df.Define("bjets_WPp85", "ZHfunctions::sel_btag(0.85)(jets_isB)")
            df = df.Define("bjets_WPp9", "ZHfunctions::sel_btag(0.9)(jets_isB)")
            df = df.Define("nbjets_WPp5", "bjets_WPp5.size()")
            df = df.Define("nbjets_WPp8", "bjets_WPp8.size()")
            df = df.Define("nbjets_WPp85", "bjets_WPp85.size()")
            df = df.Define("nbjets_WPp9", "bjets_WPp9.size()")
    
            df = df.Define("jet1", "jets_p4[0]")
            df = df.Define("jet2", "jets_p4[1]")
            df = df.Define("jet3", "jets_p4[2]")
            df = df.Define("jet4", "jets_p4[3]")
            df = df.Define("jet1_p","jet1.P()")
            df = df.Define("jet2_p","jet2.P()")
            df = df.Define("jet3_p","jet3.P()")
            df = df.Define("jet4_p","jet4.P()")
            df = df.Define("jet5_p","jets_p4.size()>4 ? jets_p4[4].P() : -999")
            df = df.Define("jet6_p","jets_p4.size()>5 ? jets_p4[5].P() : -999")
            df = df.Define("recojet_theta", "JetClusteringUtils::get_theta(jet)")
            df = df.Define("jet1_theta","recojet_theta[0]")
            df = df.Define("jet2_theta","recojet_theta[1]")
            df = df.Define("jet3_theta","recojet_theta[2]")
            df = df.Define("jet4_theta","recojet_theta[3]")
            df = df.Define("jet5_theta","jets_p4.size()>4 ? recojet_theta[4] : -999")
            df = df.Define("jet6_theta","jets_p4.size()>5 ? recojet_theta[5] : -999")
            
            df = df.Define("jet1_isTau","recojet_isTAU[0]")
            df = df.Define("jet2_isTau","recojet_isTAU[1]")
            df = df.Define("jet3_isTau","recojet_isTAU[2]")
            df = df.Define("jet4_isTau","recojet_isTAU[3]")
            df = df.Define("jet5_isTau","jets_p4.size()>4 ? recojet_isTAU[4] : -999")
            df = df.Define("jet6_isTau","jets_p4.size()>5 ? recojet_isTAU[5] : -999")
        
            df = df.Define("recojet_phi", "JetClusteringUtils::get_phi_std(jet)")
            df = df.Define("jet1_phi","recojet_phi[0]")
            df = df.Define("jet2_phi","recojet_phi[1]")
            df = df.Define("jet3_phi","recojet_phi[2]")
            df = df.Define("jet4_phi","recojet_phi[3]")
            df = df.Define("jet5_phi","jets_p4.size()>4 ? recojet_phi[4] : -999")
            df = df.Define("jet6_phi","jets_p4.size()>5 ? recojet_phi[5] : -999")
            df = df.Define("njets", "jets_p4.size()")
        
            df = df.Define("d_12", "JetClusteringUtils::get_exclusive_dmerge(_jet, 1)")
            df = df.Define("d_23", "JetClusteringUtils::get_exclusive_dmerge(_jet, 2)")
            df = df.Define("d_34", "JetClusteringUtils::get_exclusive_dmerge(_jet, 3)")
            df = df.Define("d_45", "jets_p4.size()>4 ? JetClusteringUtils::get_exclusive_dmerge(_jet, 4) : -999")
            df = df.Define("d_56", "jets_p4.size()>5 ? JetClusteringUtils::get_exclusive_dmerge(_jet, 5) : -999")


            df = df.Define("jet1_isG", "JetFlavourUtils::get_weight(MVAVec_, 0)[0]")
            df = df.Define("jet2_isG", "JetFlavourUtils::get_weight(MVAVec_, 0)[1]")
            df = df.Define("jet3_isG", "JetFlavourUtils::get_weight(MVAVec_, 0)[2]")
            df = df.Define("jet4_isG", "JetFlavourUtils::get_weight(MVAVec_, 0)[3]")
            df = df.Define("jet5_isG", "jets_p4.size()>4 ? JetFlavourUtils::get_weight(MVAVec_, 0)[4] : -999")
            df = df.Define("jet6_isG", "jets_p4.size()>5 ? JetFlavourUtils::get_weight(MVAVec_, 0)[5] : -999")
            
            df = df.Define("jet1_isQ", "JetFlavourUtils::get_weight(MVAVec_, 1)[0]")
            df = df.Define("jet2_isQ", "JetFlavourUtils::get_weight(MVAVec_, 1)[1]")
            df = df.Define("jet3_isQ", "JetFlavourUtils::get_weight(MVAVec_, 1)[2]")
            df = df.Define("jet4_isQ", "JetFlavourUtils::get_weight(MVAVec_, 1)[3]")
            df = df.Define("jet5_isQ", "jets_p4.size()>4 ? JetFlavourUtils::get_weight(MVAVec_, 1)[4] : -999")
            df = df.Define("jet6_isQ", "jets_p4.size()>5 ? JetFlavourUtils::get_weight(MVAVec_, 1)[5] : -999")
            
            df = df.Define("jet1_isS", "JetFlavourUtils::get_weight(MVAVec_, 2)[0]")
            df = df.Define("jet2_isS", "JetFlavourUtils::get_weight(MVAVec_, 2)[1]")
            df = df.Define("jet3_isS", "JetFlavourUtils::get_weight(MVAVec_, 2)[2]")
            df = df.Define("jet4_isS", "JetFlavourUtils::get_weight(MVAVec_, 2)[3]")
            df = df.Define("jet5_isS", "jets_p4.size()>4 ? JetFlavourUtils::get_weight(MVAVec_, 2)[4] : -999")
            df = df.Define("jet6_isS", "jets_p4.size()>5 ? JetFlavourUtils::get_weight(MVAVec_, 2)[5] : -999")
            
            df = df.Define("jet1_isC", "JetFlavourUtils::get_weight(MVAVec_, 3)[0]")
            df = df.Define("jet2_isC", "JetFlavourUtils::get_weight(MVAVec_, 3)[1]")
            df = df.Define("jet3_isC", "JetFlavourUtils::get_weight(MVAVec_, 3)[2]")
            df = df.Define("jet4_isC", "JetFlavourUtils::get_weight(MVAVec_, 3)[3]")
            df = df.Define("jet5_isC", "jets_p4.size()>4 ? JetFlavourUtils::get_weight(MVAVec_, 3)[4] : -999")
            df = df.Define("jet6_isC", "jets_p4.size()>5 ? JetFlavourUtils::get_weight(MVAVec_, 3)[5] : -999")
    
            df = df.Define("jet1_isB", "JetFlavourUtils::get_weight(MVAVec_, 4)[0]")
            df = df.Define("jet2_isB", "JetFlavourUtils::get_weight(MVAVec_, 4)[1]")
            df = df.Define("jet3_isB", "JetFlavourUtils::get_weight(MVAVec_, 4)[2]")
            df = df.Define("jet4_isB", "JetFlavourUtils::get_weight(MVAVec_, 4)[3]")
            df = df.Define("jet5_isB", "jets_p4.size()>4 ? JetFlavourUtils::get_weight(MVAVec_, 4)[4] : -999")
            df = df.Define("jet6_isB", "jets_p4.size()>5 ? JetFlavourUtils::get_weight(MVAVec_, 4)[5] : -999")


        return df

    # __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        #print('incl jets',jetFlavourHelper_R5.outputBranches())
        #print('excl jets',jetFlavourHelper.outputBranches())
        #all_branches += jetFlavourHelper_R5.outputBranches()
        return all_branches

        
        #all_branches+= jetClusteringHelper.outputBranches()

        ## outputs jet scores and constituent breakdown
        #branchList += jetFlavourHelper.outputBranches()
    
        #return all_branches #branchList




        ##test command fccanalysis run --nevents=10 treemaker_WbWb_reco.py
