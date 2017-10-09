import numpy as np
def load_data_training():
    """load data."""
   
    data = np.loadtxt("train.csv", dtype={'names': ('ID', 'Prediction',
                                                    'DER_mass_MMC','DER_mass_transverse_met_lep',
                                                    'DER_mass_vis','DER_pt_h,DER_deltaeta_jet_jet',
                                                    'DER_mass_jet_jet','DER_prodeta_jet_jet',
                                                    'DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt',
                                                    'DER_pt_ratio_lep_tau','DER_met_phi_centrality',
                                                    'DER_lep_eta_centrality','PRI_tau_pt','PRI_tau_eta',
                                                    'PRI_tau_phi','PRI_lep_pt','PRI_lep_eta','PRI_lep_phi',
                                                    'PRI_met','PRI_met_phi','PRI_met_sumet','PRI_jet_num',
                                                    'PRI_jet_leading_pt','PRI_jet_leading_eta',
                                                    'PRI_jet_leading_phi','PRI_jet_subleading_pt',
                                                    'PRI_jet_subleading_eta','PRI_jet_subleading_phi',
                                                    'PRI_jet_all_pt'), 
                                          'formats': (np.float, np.string_,
                                                      np.float,np.float,np.float,np.float,np.float,np.float,
                                                      np.float,np.float,np.float,np.float,np.float,np.float,
                                                      np.float,np.float,np.float,np.float,np.float,np.float,
                                                      np.float,np.float,np.float,np.float,np.float,np.float,
                                                      np.float,np.float,np.float,np.float,np.float,np.float)}
                                          , delimiter=",", skiprows=1, unpack=True)
    id_= data[0]
    y=data[1]
    x = data[2:]
    return  id_,y, x

def load_data_test():
    """load data."""
   
    data = np.loadtxt("test.csv", dtype={'names': ('ID', 'Prediction',
                                                    'DER_mass_MMC','DER_mass_transverse_met_lep',
                                                    'DER_mass_vis','DER_pt_h,DER_deltaeta_jet_jet',
                                                    'DER_mass_jet_jet','DER_prodeta_jet_jet',
                                                    'DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt',
                                                    'DER_pt_ratio_lep_tau','DER_met_phi_centrality',
                                                    'DER_lep_eta_centrality','PRI_tau_pt','PRI_tau_eta',
                                                    'PRI_tau_phi','PRI_lep_pt','PRI_lep_eta','PRI_lep_phi',
                                                    'PRI_met','PRI_met_phi','PRI_met_sumet','PRI_jet_num',
                                                    'PRI_jet_leading_pt','PRI_jet_leading_eta',
                                                    'PRI_jet_leading_phi','PRI_jet_subleading_pt',
                                                    'PRI_jet_subleading_eta','PRI_jet_subleading_phi',
                                                    'PRI_jet_all_pt'), 
                                          'formats': (np.float, np.string_,
                                                      np.float,np.float,np.float,np.float,np.float,np.float,
                                                      np.float,np.float,np.float,np.float,np.float,np.float,
                                                      np.float,np.float,np.float,np.float,np.float,np.float,
                                                      np.float,np.float,np.float,np.float,np.float,np.float,
                                                      np.float,np.float,np.float,np.float,np.float,np.float)}
                                          , delimiter=",", skiprows=1, unpack=True)
    id_= data[0]
    x = data[2:]
    return  id_, x

