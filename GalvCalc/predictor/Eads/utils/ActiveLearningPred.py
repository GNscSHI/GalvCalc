import os
import json
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.vasp.outputs import Oszicar,Vasprun
import numpy as np
from pylab import *
from pathlib import Path

from tabpfn import TabPFNRegressor
import pymatgen.core as mg
from pymatgen.core.periodic_table import Element

def target_atom_number(struct):
    for i in range(len(struct)):
        if str(struct[i]).split()[-1] == 'H':
            return i

def parse_dir_ads_energy(adsorption_structure):
    dirs, slab_H_energy_list,Eads = [],[],[]

    nn,all_distance,weight_info = [],[],[]

    distance = []
    weight = {}
    cn = VoronoiNN(tol=0, targets=None, cutoff=13.0, allow_pathological=True,
           weight='solid_angle', extra_nn_info=True)

    struct = adsorption_structure

    target_vornoi = cn.get_nn_info(struct,target_atom_number(struct))
    cnt = 1
    for v in target_vornoi:
        ele = list(v['site'].as_dict()['species'].keys())[0]
        if ele == 'H':
            continue
        distance.append(v['poly_info']['face_dist']*2)
        weight[f'{ele}_{cnt}'] = v['weight']
        cnt += 1
    nn.append(len(distance))
    all_distance.append(distance)
    weight_info.append(weight)

    save_dict = {
        'CN':nn,
        'all_distance':all_distance,
        'weight_info':weight_info
    }
        
    return pd.DataFrame(save_dict)


def generate_features(df):
    df_new = pd.DataFrame()

    # Coordination number
    df_new['CN'] = df['CN']

    # Minimum distance and average distance
    md = []
    ad = []
    for i in df['all_distance']:
        md.append(min(i))
        ad.append(np.mean(i))
    df_new['MD'] = md
    df_new['AD'] = ad

    # Weighted electronegativity
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).X
        # print()
        # res = temp/np.sum(list(weight_dict.values()))
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['EN'] = c

    # 1,2 nearest neighbor electronegativity
    temp1 = []
    temp2 = []
    for i in df['weight_info']:
        i = sorted(i.items(),key=lambda x:x[1],reverse=True)
        ele1 = i[0][0].split('_')[0]
        if len(i) > 1:
            ele2 = i[1][0].split('_')[0]
            temp2.append(Element(ele2).X)
        else:
            temp2.append(0)
        temp1.append(Element(ele1).X)

    df_new['1EN'] = temp1
    df_new['2EN'] = temp2

    # Absolute value of the difference between 1 and 2 near neighbor electronegativity
    c = []
    for i in df['weight_info']:
        i = sorted(i.items(),key=lambda x:x[1],reverse=True)
        if len(i) == 1:
            c.append(0)
        else:
            ele1 = i[0][0].split('_')[0]
            ele2 = i[1][0].split('_')[0]
            wfd = abs(Element(ele1).X-Element(ele2).X)
            c.append(wfd)
    df_new['END'] = c

    # Weight relative to atomic mass
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*(mg.Composition(ele_split).weight)
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['RAM'] = c

    # Weighted work function
    c = []
    import json
    with open(Path(__file__).parent / '../data/wf.json','r')as f:
        wf = json.load(f)
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*wf[ele_split]
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['WF'] = c

    temp1 = []
    temp2 = []
    WFD = []

    for i in df['weight_info']:

        i = sorted(i.items(),key=lambda x:x[1],reverse=True)
        ele1 = i[0][0].split('_')[0]
        temp1.append(wf[ele1])
        if len(i) > 1:
            ele2 = i[1][0].split('_')[0]
            temp2.append(wf[ele2])
            WFD.append(abs(wf[ele1]-wf[ele2]))
        else:
            temp2.append(0)
            WFD.append(0)

    df_new['WFD'] = WFD
    df_new['1WF'] = temp1
    df_new['2WF'] = temp2


    # First ionization energy
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).ionization_energies[0]
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['FIE'] = c

    # Outer atomic orbital energy
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*max(Element(ele_split).atomic_orbitals.values())
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['EOO'] = c

    # Electron affinity
    # https://calculla.com/electron_affinity
    import json
    with open(Path(__file__).parent / '../data/periodic_table_complex.json','r',encoding='utf8')as f:
        pt = json.load(f)
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*pt[ele_split]['Electron affinity']
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['EA'] = c

    # Weighted principal family and period
    g = []
    p = []

    with open(Path(__file__).parent / '../data/periodic_table.json','r',encoding='utf8')as f:
        periodic_table = json.load(f)

    for i in df['weight_info']:
        weight_dict = i
        wp = 0.0
        wg = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            wg += weight_dict[ele]*periodic_table[ele_split]['group']
            wp += weight_dict[ele]*int(periodic_table[ele_split]['period'])
        res_wg = wg/sum(list(weight_dict.values()))
        res_wp = wp/sum(list(weight_dict.values()))
        g.append(res_wg)
        p.append(res_wp)
    df_new['GN'] = g
    df_new['PN'] = p

    # Weighted atomic number
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).Z
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)

    df_new['AN'] = c

    # resistance
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).electrical_resistivity
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['ER'] = c

    # Atomic radius
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).atomic_radius
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['AR'] = c

    # Thermal conductivity
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).thermal_conductivity
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['TC'] = c

    # Melting point and boiling point
    tmp1 = []
    tmp2 = []
    for i in df['weight_info']:
        weight_dict = i
        temp1,temp2 = 0.0, 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp1 += weight_dict[ele]*Element(ele_split).melting_point
            temp2 += weight_dict[ele]*Element(ele_split).boiling_point
        res1 = temp1/sum(list(weight_dict.values()))
        res2 = temp2/sum(list(weight_dict.values()))
        tmp1.append(res1)
        tmp2.append(res2)
    df_new['MP'] = tmp1
    df_new['BP'] = tmp2

    # Bulk modulus
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            try:
                temp += weight_dict[ele]*Element(ele_split).bulk_modulus
            except:
                temp += 0
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['BM'] = c

    # Moore volume
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).molar_volume
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['MV'] = c

    # Oxidation state 
    import json
    with open(Path(__file__).parent / '../data/periodic_table.json','r',encoding='utf8')as f:
        pt = json.load(f)
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            try:
                temp += weight_dict[ele]*pt[ele_split]['ICSD oxidation states'][-1]
            except:
                temp += 0 # For the elements that don't have an oxidation state, let's do it at zero
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['OS'] = c

    # Unoccupied electron number (UOE)
    import json
    with open(Path(__file__).parent / '../data/uoe.json','r',encoding='utf8')as f:
        pt = json.load(f)

    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*pt[ele_split]
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['UOE'] = c
    
    return df_new


def predict(adsorption_structures):
    """
    Predict adsorption energies using TabPFN.

    Parameters
    ----------
    adsorption_structures : Structure or list[Structure]
        A pymatgen Structure object or a list of Structure objects.

    Returns
    -------
    float or np.ndarray
        Predicted adsorption energy(s).
    """

    single_input = False
    if not isinstance(adsorption_structures, (list, tuple)):
        adsorption_structures = [adsorption_structures]
        single_input = True

    df_list = []

    for struct in adsorption_structures:
        df_list.append(parse_dir_ads_energy(struct))

    df = pd.concat(df_list, ignore_index=True)

    df_new = generate_features(df)

    train_df = pd.read_excel(
        Path(__file__).parent / "../data/trainset_loop7.xlsx"
    )

    train_df = train_df.drop(["PN", "AN"], axis=1)

    y_train = train_df["Eads"].values
    X_train = train_df.drop(columns=["Eads"])

    features = [
        'CN', 'MD', 'AD', 'EN', '1EN', '2EN', 'END',
        'RAM', 'WF', 'WFD', '1WF', '2WF',
        'FIE', 'EOO', 'EA', 'GN',
        'ER', 'AR', 'TC', 'MP', 'BP',
        'BM', 'MV', 'OS', 'UOE'
    ]

    X_test = df_new[features]
    model = TabPFNRegressor(
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if hasattr(y_pred, "detach"):  # torch.Tensor
        y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred + 0.19

    if single_input:
        return float(y_pred[0])

    return y_pred
