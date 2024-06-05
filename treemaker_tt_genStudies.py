import os, copy, ROOT

# list of processes
processList = {
    "wzp6_ee_WbWb_semihad_ecm345": {
        "fraction": 1,
        "crossSection": 1,
    },
    
    "wzp6_ee_WbWb_semihad_ecm350": {
        "fraction": 1,
        "crossSection": 1,
    },

    "wzp6_ee_WbWb_semihad_ecm355": {
        "fraction": 1,
        "crossSection": 1,
    },
    "wzp6_ee_WbWb_semihad_ecm340":{
        "fraction": 1,
        "crossSection": 1,
    },
}

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
#prodTag     = "FCCee/winter2023/IDEA/"

#Optional: output directory, default is local running directory
outputDir   = "./outputs/treemaker/tt/acceptance_atleast1_genlep_status1"


# additional/costom C++ functions, defined in header files (optional)
includePaths = ["examples/functions.h"]

## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc_v1"

## model files needed for unit testing in CI

## model files locally stored on /eos
eos_dir ="/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"


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

#files_In=get_files(eos_dir,samples[0])
inputDir    = eos_dir#get_files(eos_dir,samples[0])
#from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)

#jetFlavourHelper = None
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)

jetClusteringHelper = None

# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:

    # __________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):

        # __________________________________________________________
        # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2

        # define some aliases to be used later on
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")

        # get all the leptons from the collectio
        df = df.Define("status1parts",           "FCCAnalyses::MCParticle::sel_genStatus(1)(Particle)")
        df = df.Define("gen_leps_status1",       "FCCAnalyses::MCParticle::sel_genleps(13,11, true)(status1parts)")
        df = df.Define("gen_neutrinos_status1",  "FCCAnalyses::MCParticle::sel_genleps(14,12, true)(status1parts)")
        df = df.Define("ngen_leps_status1",      "FCCAnalyses::MCParticle::get_n(gen_leps_status1)")        
        df = df.Define("ngen_neutrinos_status1", "FCCAnalyses::MCParticle::get_n(gen_neutrinos_status1)")
        
        df = df.Define("gen_leps_status1_p",      "FCCAnalyses::MCParticle::get_p(gen_leps_status1)")
        df = df.Define("gen_leps_status1_theta",  "FCCAnalyses::MCParticle::get_theta(gen_leps_status1)")
        df = df.Define("gen_leps_status1_phi",    "FCCAnalyses::MCParticle::get_phi(gen_leps_status1)")
        df = df.Define("gen_leps_status1_charge", "FCCAnalyses::MCParticle::get_charge(gen_leps_status1)")
        df = df.Define("gen_leps_status1_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_leps_status1)")
        df = df.Define("gen_leps_status1_mother_pdgId", "FCCAnalyses::MCParticle::get_leptons_origin(gen_leps_status1,Particle,Particle0)")

        df = df.Define("gen_neutrinos_status1_p",      "FCCAnalyses::MCParticle::get_p(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_theta",  "FCCAnalyses::MCParticle::get_theta(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_phi",    "FCCAnalyses::MCParticle::get_phi(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_charge", "FCCAnalyses::MCParticle::get_charge(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_pdgId",  "FCCAnalyses::MCParticle::get_pdg(gen_neutrinos_status1)")
        df = df.Define("gen_neutrinos_status1_mother_pdgId", "FCCAnalyses::MCParticle::get_leptons_origin(gen_neutrinos_status1,Particle,Particle0)")
        
        df = df.Define('gen_lightquarks',    'FCCAnalyses::MCParticle::sel_lightQuarks(true)(Particle)')
        df = df.Define("ngen_partons","FCCAnalyses::MCParticle::get_n(gen_lightquarks)");
        df = df.Filter("ngen_leps_status1 > 0")


        
        
        #print('this',df["gen_lep_status1_p"].to_string(index=False))        
        df = df.Define(
            "GenParticlesNoLeps",
            "FCCAnalyses::MCParticle::remove(status1parts,gen_leps_status1)",
        )
        df = df.Define(
            "GenParticlesNoLepsNoNeu",
            "FCCAnalyses::MCParticle::remove(GenParticlesNoLeps,gen_neutrinos_status1)",
        )
        #global jetClusteringHelper
        
        df = df.Define("GP_px",   "MCParticle::get_px(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_py" ,  "MCParticle::get_py(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_pz" ,  "MCParticle::get_pz(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_e"  ,  "MCParticle::get_e(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_m"  ,  "MCParticle::get_mass(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_q"  ,  "MCParticle::get_charge(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_p"  ,  "MCParticle::get_p(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_pdgId"  ,  "MCParticle::get_pdg(GenParticlesNoLepsNoNeu)")
        df = df.Define("GP_status"  ,  "MCParticle::get_genStatus(GenParticlesNoLepsNoNeu)")
        
        df = df.Define("pseudo_jets",         "JetClusteringUtils::set_pseudoJets_xyzm(GP_px, GP_py, GP_pz, GP_m)")
        df = df.Define("FCCAnalysesJets_kt",  "JetClustering::clustering_kt(0.5, 2, 4, 0, 20)(pseudo_jets)")
        df = df.Define("Genjets_kt",          "JetClusteringUtils::get_pseudoJets(FCCAnalysesJets_kt)")
        df = df.Define("Genjets_kt_e",        "JetClusteringUtils::get_e(Genjets_kt)")
        df = df.Define("Genjets_kt_px",       "JetClusteringUtils::get_px(Genjets_kt)")
        df = df.Define("Genjets_kt_py",       "JetClusteringUtils::get_py(Genjets_kt)")
        df = df.Define("Genjets_kt_pz",       "JetClusteringUtils::get_pz(Genjets_kt)")
        df = df.Define("Genjets_kt_m",        "JetClusteringUtils::get_m(Genjets_kt)")


        
        return df

    # __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
            "ngen_partons",
            "ngen_leps_status1", 
            "gen_leps_status1_p",      
            "gen_leps_status1_theta",  
            "gen_leps_status1_phi",    
            "gen_leps_status1_charge", 
            "gen_leps_status1_pdgId",  "gen_leps_status1_mother_pdgId","GP_p","GP_pdgId","GP_status",
            "Genjets_kt_e",
            "Genjets_kt_px",
            "Genjets_kt_py",
            "Genjets_kt_pz",
            "Genjets_kt_m"
        ]


        ## outputs jet scores and constituent breakdown
        #branchList += jetFlavourHelper.outputBranches()

        return branchList
