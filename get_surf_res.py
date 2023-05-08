#!/home/sagardip/.conda/envs/sdp-pyrosetta/bin/python

import sys
import glob
import os
import argparse
from pyrosetta import *
from rosetta import *
from pyrosetta.rosetta.core.select.residue_selector import *

init(
    "-beta_nov16"
    + " -holes:dalphaball /work/tlinsky/Rosetta/main/source/external/DAlpahBall/DAlphaBall.macgcc"
    + " -use_terminal_residues true -mute basic.io.database core.scoring"
)

#xml = "/home/sagardip/protocols/relax_sym.xml"
#objs = protocols.rosetta_scripts.XmlObjects.create_from_file(xml)
#makepolyA = objs.get_mover("makepolyA")



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('cleanpdbs', help="Path to opm PDBs whose surface TMD residues are to be appended", action='store', type=str)

    parser.add_argument('--pdbs', action='store', type=str, required=True)

    parser.add_argument('--TMD', help="H or E depending on helical or beta sheet topologies of the transmembrane domain", action='store', type=str, required=True)
    
    args = parser.parse_args()

    return args


args = get_args()


for pdbin in glob.glob(args.cleanpdbs+"*.clean.pdb"):
    poseinp = pose_from_file(pdbin)
    if args.TMD == "H":
        core_selector = LayerSelector()
        core_selector.set_angle_exponent(1.0)
        core_selector.set_angle_shift_factor(0.5)
        core_selector.set_dist_exponent(0.5)
        core_selector.set_sc_neighbor_dist_midpoint(18.0)
        core_selector.set_cutoffs(25.0, 2.0)
        core_selector.set_layers(True,False,False)
        corebool = core_selector.apply(poseinp)
        surf_res = [i for i in range(1, poseinp.size()+1) if not corebool[i]]
    if args.TMD == "E":
        core_selector = LayerSelector()
        if poseinp.size() < 300:
            core_selector.set_angle_exponent(1.0)
            core_selector.set_angle_shift_factor(0.8)
            core_selector.set_dist_exponent(0.2)
            core_selector.set_sc_neighbor_dist_midpoint(30.0)
            core_selector.set_cutoffs(40.0, 2.0)
        else:
            core_selector.set_angle_exponent(0.95)
            core_selector.set_angle_shift_factor(0.75)
            core_selector.set_dist_exponent(0.2)
            core_selector.set_sc_neighbor_dist_midpoint(150.0)
            core_selector.set_cutoffs(200.0, 2.0)
        core_selector.set_layers(True,False,False)
        corebool = core_selector.apply(poseinp)
        surf_res = [i for i in range(1, poseinp.size()+1) if not corebool[i]]
    with open(pdbin, 'r+') as fout:
        for linein in fout:
            pass
        with open('/'.join(args.pdbs.split('/')) + '/' + pdbin.split('/')[-1].replace(".clean.pdb",".pdb"), 'r') as fin:
            count = 0
            for line in fin:
                l = line.strip().split()
                if line[:4] == "HETA" and "DUM" in line  and count == 0:
                    zlow = float(l[-1])
                    count += 1
                elif "DUM" in line and count == 1:
                    zhigh = float(l[-1])
                    if zlow > zhigh:
                        zlow, zhigh = zhigh, zlow
                    count += 1
                else:
                    pass
        TMD_res = [poseinp.pdb_info().pose2pdb(i).strip() for i in surf_res if poseinp.residue(i).xyz('CA')[2] <= zhigh and poseinp.residue(i).xyz('CA')[2] >= zlow] 
        
        fout.write("\n" + "TM_domain_surface_residues," + ",".join(TMD_res))


