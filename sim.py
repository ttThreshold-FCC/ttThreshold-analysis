import ROOT
ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)
f1=ROOT.TFile.Open("outputs/treemaker/WbWb/semihad/wzp6_ee_WbWb_ecm345.root");
f2=ROOT.TFile.Open("outputs/treemaker/WbWb/semihad/wzp6_ee_WWZ_Zbb_ecm345.root")
t1=f1.Get("events")
t2=f2.Get("events")
print(type(t1),type(t2))
hist1   = ROOT.TH1F("hist1" ,'',50,0,250)
hist2   = ROOT.TH1F("hist2" ,'',50,0,250)
t1.Draw('mbbar>>hist1', '', 'goff');     hist1.SetDirectory(0)
t2.Draw('mbbar>>hist2', '', 'goff');     hist2.SetDirectory(0)
canv = ROOT.TCanvas('canv', 'bar', 600, 600)
canv.cd()

leg = ROOT.TLegend(0.6, 0.82, 0.75, 0.89)
leg.SetTextSize(0.03)
leg.SetFillStyle(0)
leg.AddEntry(hist1, 'WbWb'  , 'l')
leg.AddEntry(hist2, 'WWZ'  , 'l')


hist1.SetLineColor(ROOT.kRed)
hist2.SetLineColor(ROOT.kBlue)
hist1.GetYaxis().SetTitle('a.u.')
hist1.GetXaxis().SetTitle("m_{b#bar{b}}")
hist1.Scale(1.0/hist1.Integral())
hist2.Scale(1.0/hist2.Integral())
hist1.GetYaxis().SetRangeUser(0,0.22)
hist1.Draw('hist')
hist2.Draw('histsame')
leg.SetLineColor(ROOT.kWhite)
leg.Draw('same')
canv.SaveAs('/eos/user/a/anmehta/www//FCC_top/mbbar.pdf')
canv.SaveAs('/eos/user/a/anmehta/www//FCC_top/mbbar.png')
