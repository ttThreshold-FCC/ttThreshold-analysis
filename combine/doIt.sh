#!/bin/bash
for ecm in  365  #340 365 #340 345 365 #350 355 345 365 340
do
#    for channel in semihad had
#    do
#	for var in njets_R5 nbjets_R5_eff_p9 mbbar jet1_R5_p jet2_R5_p jet1_R5_theta jet2_R5_theta  jet3_R5_p jet3_R5_theta jet4_R5_p jet4_R5_theta #lep_p lep_theta jet5_R5_p jet5_R5_theta jet6_R5_p jet6_R5_theta #nbjets_R5_true njets_R5
#	do
#	    python plotter.py  -i --vname ${var} --sel no_cut  -p  -c ${channel}  -e ${ecm}   --style norm  
#	done
#    done 

    #    for channel in semihad had #lep
    #   do
	#python plotter_withsyst.py  --vname  singlebin --sel zerob  -c ${channel} -e ${ecm}  -p --logy
	#python plotter_withsyst.py  --vname  singlebin --sel zerob  -c ${channel} -e ${ecm}    -p 
	#python plotter_withsyst.py  --vname  njets_R5   --sel oneb    -c ${channel} -e ${ecm}  -p    --logy
	#python plotter_withsyst.py  --vname  njets_R5   --sel zerob   -c ${channel} -e ${ecm}  -p    --logy
	#python plotter_withsyst.py  --vname  njets_R5   --sel twob    -c ${channel} -e ${ecm}  -p    --logy
	#python plotter_withsyst.py  --vname  njets_R5   --sel no_cut  -c ${channel} -e ${ecm}  -p    --logy
	
	#python plotter_withsyst.py  --vname  njets_R5   --sel oneb    -c ${channel} -e ${ecm}  -p # -i
	#python plotter_withsyst.py  --vname  njets_R5   --sel zerob   -c ${channel} -e ${ecm}  -p # -i
	#python plotter_withsyst.py  --vname  njets_R5   --sel twob    -c ${channel} -e ${ecm}  -p # -i 
	#python plotter_withsyst.py  --vname  njets_R5   --sel no_cut  -c ${channel} -e ${ecm}  -p # -i

#	python plotter_withsyst.py  --vname  BDT_score --sel twob --msel njgt0  -c semihad -e ${ecm}  -p 
#	python plotter_withsyst.py  --vname  BDT_score --sel twob --msel njgt1  -c had     -e ${ecm}  -p  # --logy
#
#
	python plotter_withsyst.py  --vname  singlebin  --sel zerob   --msel njgt0 -c  semihad -e ${ecm}  -p  --logy 
#	python plotter_withsyst.py  --vname  njets_R5   --sel oneb    --msel njgt0 -c  semihad -e ${ecm}  -p -i --logy
#	python plotter_withsyst.py  --vname  njets_R5   --sel twob    --msel njgt0 -c  semihad -e ${ecm}  -p -i --logy
	python plotter_withsyst.py  --vname  singlebin  --sel zerob   --msel njgt0 -c  semihad -e ${ecm}  -p  
#	python plotter_withsyst.py  --vname  njets_R5   --sel oneb    --msel njgt0 -c  semihad -e ${ecm}  -p -i 
#	python plotter_withsyst.py  --vname  njets_R5   --sel twob    --msel njgt0 -c  semihad -e ${ecm}  -p -i 
    
#	python plotter_withsyst.py  --vname  njets_R5   --sel oneb    --msel njgt1 -c  had -e ${ecm}  -p  -i --logy
	#python plotter_withsyst.py  --vname  njets_R5   --sel zerob   --msel njgt1 -c  had -e ${ecm}  -p   --logy
#	python plotter_withsyst.py  --vname  njets_R5   --sel twob    --msel njgt1 -c  had -e ${ecm}  -p  -i --logy
#	python plotter_withsyst.py  --vname  njets_R5   --sel oneb    --msel njgt1 -c  had -e ${ecm}  -p  -i 
	#python plotter_withsyst.py  --vname  njets_R5   --sel zerob   --msel njgt1 -c  had -e ${ecm}  -p  -i 
#	python plotter_withsyst.py  --vname  njets_R5   --sel twob    --msel njgt1 -c  had -e ${ecm}  -p  -i  
    
	#python plotter_withsyst.py  --vname  njets_R5   --sel no_cut  --msel njgt0 -c  ${channel} -e ${ecm}  -p    --logy
	#python plotter_withsyst.py  --vname  njets_R5   --sel zerob   --msel njgt0 -c  ${channel} -e ${ecm}  -p    --logy
	#python plotter_withsyst.py  --vname  njets_R5   --sel zerob   --msel njgt0 -c  ${channel} -e ${ecm}  -p  #-i
	#python plotter_withsyst.py  --vname  njets_R5   --sel no_cut  --msel njgt0 -c  ${channel} -e ${ecm}  -p  #-i
	#python plotter_withsyst.py  --vname  BDT_score --sel twob   --msel njgt0 -c  ${channel} -e ${ecm}  -p 
	#python plotter_withsyst.py  --vname  BDT_score --sel twob   --msel njgt0 -c  ${channel} -e ${ecm}  -p   --logy
	

	    
	    #python plotter_withsyst.py  --vname  singlebin --sel zerob   --msel njgt1 -c  ${channel} -e ${ecm}  -p --logy
	    #python plotter_withsyst.py  --vname  singlebin --sel zerob  --msel njgt1 -c  ${channel} -e ${ecm}    -p -i
	    
	   # python plotter_withsyst.py  --vname  njets_R5   --sel no_cut  --msel njgt1 -c  ${channel} -e ${ecm}  -p    --logy
	    
	    #python plotter_withsyst.py  --vname  njets_R5   --sel no_cut  --msel njgt1 -c  ${channel} -e ${ecm}  -p  -i

	    #python plotter_withsyst.py  --vname  BDT_score --sel twob   --msel njgt1 -c  ${channel} -e ${ecm}  -p 
	    #python plotter_withsyst.py  --vname  BDT_score --sel twob   --msel njgt1 -c  ${channel} -e ${ecm}  -p   --logy

	    
	    #python plotter_withsyst.py  --vname  nbjets_R5_eff_p9 --sel no_cut   -c ${channel} -e ${ecm}  -p 
	    #python plotter_withsyst.py  --vname  nbjets_R5_eff_p9 --sel no_cut   -c ${channel} -e ${ecm}  -p   --logy

	    

	    #python plotter_withsyst.py  --vname ${var} --sel zerob  -c ${channel} -e ${ecm} #-p 
	    #python plotter_withsyst.py  --vname ${var} --sel oneb   -c ${channel} -e ${ecm} #-p 
	    #python plotter_withsyst.py  --vname ${var} --sel twob   -c ${channel} -e ${ecm} #-p

	    #python plotter_withsyst_singleNP.py  --vname ${var} --sel zerob  -c ${channel} -e ${ecm}
	    #python plotter_withsyst_singleNP.py  --vname ${var} --sel oneb   -c ${channel} -e ${ecm}
	    #python plotter_withsyst_singleNP.py  --vname ${var} --sel twob   -c ${channel} -e ${ecm}

	    #python plotter.py   --vname BDT_score --sel oneb  -p  -c ${channel}  -e ${ecm}   --style norm #--logy
	    #python plotter.py   --vname BDT_score --sel zerob  -p  -c ${channel}  -e ${ecm}  --style norm #--logy
	    #python plotter.py   --vname BDT_score --sel twob  -p  -c ${channel}  -e ${ecm}   --style norm #--logy 
	    #python plotter.py   --vname ${var} --sel zerob   -p  -c ${channel}  -e ${ecm}   --style norm
	    #python plotter.py   --vname ${var} --sel twob    -p  -c ${channel}  -e ${ecm}   --style norm 
	    #python plotter.py   --vname ${var} --sel oneb    -p  -c ${channel}  -e ${ecm}   --style norm
  #  done
    done	
#done 
