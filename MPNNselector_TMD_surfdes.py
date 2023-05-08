#!/home/sagardip/.conda/envs/sdp-pyrosetta/bin/python

import sys
import glob
import os
import numpy as np
import argparse
import json
from prody import *
from pyrosetta import *
from rosetta import *
from pyrosetta.rosetta.core.select.residue_selector import *
import shutil as sh
init()


chainstr = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

AA_list = 'ACDEFGHIKLMNPQRSTVWY'

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pdbs', help="Path to pdbs whose secondary structure elements are to be designed with MPNN", action='store', type=str, required=True)

    parser.add_argument('--ss', help="Comma separated string of secondary structue elements to be designed, e.g E,L,H or E,L", action='store', type=str, default='H')
    
    parser.add_argument('--diffused', help="If true, all glycine positions with 3 or more consecutive glycines will be designed in the provided diffused chain id", action='store', type=str, default=None)

    parser.add_argument('--nseq', help="Number of MPNN sequences per input graft", action='store', type=int, default=32)

    parser.add_argument('--bs', help="Batch size for MPNN", action='store', type=int, default=8)
    
    parser.add_argument('--interface', help="Specify if you want to design chain1 interface. Can be coupled with secondary structure selection", action='store', type=bool, default=False)
    
    parser.add_argument('--symm', help="Bool for symetric pose. Must be true when MPNNing symmetric pose", action='store', type=bool, default=False)

    parser.add_argument('--inpaint', help="If inpaint true, residue indices will be selected from corresponding trb files", action='store', type=bool, default=False)
    
    parser.add_argument('--mem', help="Memory for gpu", action='store', type=str, default='16g')
    
    parser.add_argument('--resi', help="Residue string for design", action='store', type=str, default='0')
    
    parser.add_argument('--surface', help="Include neighbor residues for surface design of alpha helical proteins. default distance is 5.6 A", action='store', type=bool, default=False)
    
    parser.add_argument('--TMdepth', help="Depth of transmembrane domain", action='store', type=float, default=40.0)
   
    parser.add_argument('--fixed', help="Residue indices to be kept fixed", action='store', type=str, default=None)
    parser.add_argument('--interf_frac', help="Percent of transmembrane domain to be designed with Tyr Trp for demarcating lipid headgoup interactions", action='store', type=float, default=0.25)
    
    parser.add_argument('--pore', help="Pore residue indexes to be kept fixed in case of alpha-helical TM pore, only provide for one chain for a helical TM oligomer. Will be automatically set to the core residues for a beta barrel TM protein/oligomer and designed with non-hydrophobics, so not required but can be specified in which case will override core selection and design non-chosen core residues with all amino acids", action='store', type=str, default=None)
    parser.add_argument('--poreseq', help="Pore AA to be allowed only for designing", action='store', type=str, default='DEGHKLMNPQRSTYW')
    parser.add_argument('--buriedsurfseq', help="Buried surface AA to be allowed only for designing", action='store', type=str, default='AFGILV')
    parser.add_argument('--intfsurfseq', help="Interface surface AA to be allowed only for designing", action='store', type=str, default='LFIAYW')
    
    parser.add_argument('--cis_resid', help="Residue index of chainA of pdb which should preferably be on the cis side of the TM domain", action='store', type=str, default='0')

    parser.add_argument('--gpu', help="Type of gpu", action='store', type=str, default='a4000')
    
    parser.add_argument('--num_packs', help="Number of packed pdbs per sequence", action='store', type=str, default='1')
    parser.add_argument('--tmdesign', help="Specify if MPNN trained with global transmembrane label should be used", action='store', type=str, default="0")
    parser.add_argument('--outdir', help="Output directory", action='store', type=str, default='./')
    parser.add_argument('--TMD', help="Specify if MPNN trained with specific residue types of transmembrane labels (interface and buried) should be used", action='store', type=bool, default=False)
    args = parser.parse_args()

    return args


def writeTask(file, fres, folder, Nseq, Nbatch, tiedres=[], ch="A", buried=None, intf=None):
    file.write(f'chain_list_input="{ch}"\n\nfolder_with_pdbs="{folder}"\npath_for_parsed_chains="{folder+"/parsed_pdbs.jsonl"}"\npath_for_assigned_chains="{folder+"/assigned_pdbs.jsonl"}"\npath_for_fixed_positions="{[folder+"/fixed_pos.jsonl" if fres else "[]" for i in range(1)][0]}"\npath_for_tied_positions="{[folder+"/tied_pdbs.jsonl" if tiedres else "[]" for i in range(1)][0]}"\nfixed_positions="{fres}"\ntied_positions="{tiedres}"\npath_for_designed_sequences="{folder+"/temp_0.1"}"\n\n')
    file.write('python /home/sagardip/MPNN_scPack/proteinmpnn/helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains\n\n')
    file.write('python /home/sagardip/MPNN_scPack/proteinmpnn/helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chain_list_input"\n\n')
    if fres:
        file.write('python /home/sagardip/MPNN_scPack/proteinmpnn/helper_scripts/make_fixed_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_fixed_positions --position_list "$fixed_positions" --chain_list "$chain_list_input"\n')
    if tiedres:
        file.write('python /home/sagardip/MPNN_scPack/proteinmpnn/helper_scripts/make_tied_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_tied_positions --chain_list "$chain_list_input" --position_list "$tied_positions"\n\n')

    if args.tmdesign == "yes":
        file.write(f'python /home/sagardip/MPNN_scPack/proteinmpnn/protein_mpnn_run.py --jsonl_path $path_for_parsed_chains --fixed_positions_jsonl $path_for_fixed_positions --chain_id_jsonl $path_for_assigned_chains --tied_positions_jsonl $path_for_tied_positions --out_folder $path_for_designed_sequences --num_seq_per_target {Nseq} --sampling_temp "0.1" --batch_size {Nbatch} --pack_side_chains 1 --num_packs {args.num_packs} --transmembrane yes\n\n\n')
    elif args.tmdesign == "no":
        file.write(f'python /home/sagardip/MPNN_scPack/proteinmpnn/protein_mpnn_run.py --jsonl_path $path_for_parsed_chains --fixed_positions_jsonl $path_for_fixed_positions --chain_id_jsonl $path_for_assigned_chains --tied_positions_jsonl $path_for_tied_positions --out_folder $path_for_designed_sequences --num_seq_per_target {Nseq} --sampling_temp "0.1" --batch_size {Nbatch} --pack_side_chains 1 --num_packs {args.num_packs} --transmembrane no\n\n\n')
    elif args.tmdesign == "0" and not args.TMD:
        file.write(f'python /home/sagardip/MPNN_scPack/proteinmpnn/protein_mpnn_run.py --jsonl_path $path_for_parsed_chains --fixed_positions_jsonl $path_for_fixed_positions --chain_id_jsonl $path_for_assigned_chains --tied_positions_jsonl $path_for_tied_positions --out_folder $path_for_designed_sequences --num_seq_per_target {Nseq} --sampling_temp "0.1" --batch_size {Nbatch} --pack_side_chains 1 --num_packs {args.num_packs}\n\n\n')
    elif args.TMD:
        file.write(f'python /home/sagardip/MPNN_scPack/proteinmpnn/protein_mpnn_run.py --jsonl_path $path_for_parsed_chains --fixed_positions_jsonl $path_for_fixed_positions --chain_id_jsonl $path_for_assigned_chains --tied_positions_jsonl $path_for_tied_positions --out_folder $path_for_designed_sequences --num_seq_per_target {Nseq} --sampling_temp "0.1" --batch_size {Nbatch} --pack_side_chains 1 --num_packs {args.num_packs} --use_seed 1 --seed 17 --transmembrane_chain_ids "{",".join(ch.split())}" --transmembrane_buried "{buried}" --transmembrane_interface "{intf}"\n\n\n')


def orient_pdb(pdbin, pose, resi):
    if "cterm" in resi:
        chA = pose.split_by_chain(1)
        resi = chA.size()
    if pose.residue(int(resi)).xyz('CA')[2] > 0:
        index = utility.vector1_bool()
        for i in range(1, pose.size()+1):
            index.append(1)
        protocols.toolbox.pose_manipulation.rigid_body_move(numeric.xyzVector_double_t(1,0,0),180,numeric.xyzVector_double_t(0,0,0),pose,index)

        pdbname = pdbin[:-4] + '_oriented.pdb'

        pose.dump_pdb(pdbname)
        st = parsePDB(pdbin[:-4]+"_oriented.pdb", subset='ca')
        coords = st.getCoords()
    else:
        st = parsePDB(pdbin, subset='ca')
        coords = 0
        pdbname = pdbin

    return st, coords, pdbname

def find_intf_buried_res(st, tmd, intf_frac):

    coords = st.getCoords()
    
    maxZ = np.max(coords, 0)[2]
    minZ = np.min(coords, 0)[2]

    span = maxZ - minZ

    if span > tmd:
        resi_tmd = 1 + np.nonzero(np.where(coords[:,2]< minZ+tmd, coords[:,2], 0))[0]
        maxZ = np.max(coords[resi_tmd-1,2])
        span = maxZ - minZ
    else:
        resi_tmd = st.getResnums()

    intf = (span*intf_frac)/2

    intf1_tmd = [str(x+1) for x in np.where(coords[resi_tmd-1,2]<= minZ+1.5*intf, resi_tmd-1, 0) if x != 0.0]

    intf2_tmd = [str(x+1) for x in np.where(coords[resi_tmd-1,2]>= maxZ-intf, resi_tmd-1, 0) if x != 0.0]

    buried1_tmd = [str(x+1) for x in np.where(coords[resi_tmd-1,2]> minZ+1.5*intf, resi_tmd-1, 0) if x != 0.0]

    buried2_tmd = [str(x+1) for x in np.where(coords[resi_tmd-1,2]< maxZ-intf, resi_tmd-1, 0) if x != 0.0]

    buriedtmd = [str(y) for y in resi_tmd if str(y) in buried1_tmd and str(y) in buried2_tmd]

    return intf1_tmd+intf2_tmd, buriedtmd

def final_surf_res_for_design(pose, tmd, intftmd, buriedtmd, dssp):
    sym = 0
    if pose.num_chains() > 1:
        chainA = pose.split_by_chain(chain_id=1)
        sym = 1
    else:
        chainA = pose.clone()

    if tmd == "H":
        core_selector = LayerSelector()
        core_selector.set_angle_exponent(1.0)
        core_selector.set_angle_shift_factor(0.5)
        core_selector.set_dist_exponent(0.5)
        core_selector.set_sc_neighbor_dist_midpoint(18.0)
        core_selector.set_cutoffs(25.0, 2.0)
        core_selector.set_layers(True,False,False)
        corebool = core_selector.apply(pose)
        surf_res = [str(i) for i in range(1, chainA.size()+1) if not corebool[i]]

        intfinit = [j for j in intftmd if j in surf_res]
        buriedinit = [j for j in buriedtmd if j in surf_res]


        intfinit_selector = ResidueIndexSelector()
        intfinit_selector.set_index(",".join(intfinit))
        buriedinit_selector = ResidueIndexSelector()
        buriedinit_selector.set_index(",".join(buriedinit))

        #intftmdneighbors = NeighborhoodResidueSelector()
        #intftmdneighbors.set_distance(5.6)
        #intftmdneighbors.set_include_focus_in_subset(True)
        #intftmdneighbors.set_focus_selector(intfinit_selector)
        #intftmd_bool = intftmdneighbors.apply(chainA)
        intftmd_bool = [True if str(x) in intfinit else False for x in range(1, chainA.size()+1)]

        #buriedtmdneighbors = NeighborhoodResidueSelector()
        #buriedtmdneighbors.set_distance(5.6)
        #buriedtmdneighbors.set_include_focus_in_subset(True)
        #buriedtmdneighbors.set_focus_selector(buriedinit_selector)
        #buriedtmd_bool = buriedtmdneighbors.apply(chainA)
        buriedtmd_bool = [True if str(x) in buriedinit else False for x in range(1, chainA.size()+1)]

        if sym == 1:
            chain1 = ChainSelector("1")
            chain2 = ChainSelector("2")
            chainend = ChainSelector(f'{pose.num_chains()}')
            interface_selector1 = InterGroupInterfaceByVectorSelector( chain1, chain2 )
            interface_selector2 = InterGroupInterfaceByVectorSelector( chain1, chainend )
            interface_selector = OrResidueSelector(interface_selector1, interface_selector2)
            interface_chain1 = AndResidueSelector(chain1, interface_selector)
            interface_bool = interface_chain1.apply(pose)

            intf_final = [str(i) for i in range(1, chainA.size()+1) if intftmd_bool[i-1] and not interface_bool[i] and dssp[i-1]==tmd]

            buried_final = [str(j) for j in range(1, chainA.size()+1) if buriedtmd_bool[j-1] and not interface_bool[j] and dssp[j-1]==tmd and j not in intf_final]
        else:
            intf_final = [str(i) for i in range(1, chainA.size()+1) if intftmd_bool[i-1] and dssp[i-1]==tmd]
            buried_final = [str(j) for j in range(1, chainA.size()+1) if buriedtmd_bool[j-1] and dssp[j-1]==tmd and j not in intf_final]

        pore_final = [str(i) for i in range(1, chainA.size()+1) if corebool[i]]


    elif tmd == "E":
        core_selector = LayerSelector()
        if pose.size() < 300:
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
        surf_res = [i for i in range(1, chainA.size()+1) if not corebool[i]]
        intf_final = [int(j) for j in intftmd if j in surf_res and dssp[j-1] == "E"]
        buried_final = [int(j) for j in buriedtmd if j in surf_res and dssp[j-1] == "E"]
        pore_final = [i for i in range(1, chainA.size()+1) if corebool[i] and dssp[i-1] == "E"]

    return pore_final, intf_final, buried_final



def make_omit_AA_json(pdbname, st, pore, poreseq, intfall, intfseq, buriedall, buriedseq,  outfolder):
    final_dict = {}
    my_dict = {}
    nchains = st.numChains()
    chids = st.getChids()
    if nchains > 1:
        l = int(len(st)/nchains)
        for j in range(0,nchains):
            if pore:
                my_dict[chids[l*j+1]] = [[pore, poreseq], [intfall, intfseq], [buriedall, buriedseq]]
            else:
                my_dict[chids[l*j+1]] = [[intfall, intfseq], [buriedall, buriedseq]]
        final_dict[pdbname] = my_dict
    else:
        if pore:
            my_dict[chids[1]] = [[pore, poreseq], [intfall, intfseq], [buriedall, buriedseq]]
        else:
            my_dict[chids[1]] = [[intfall, intfseq], [buriedall, buriedseq]]
        final_dict[pdbname] = my_dict

    with open(outfolder+'/omit_AA.jsonl', 'w') as fout:
        fout.write(json.dumps(final_dict))

    return outfolder+'/omit_AA.jsonl'



args = get_args()
ss = "".join(args.ss.split(","))
if "-" in args.resi:
    t = args.resi.split("-")
    resi = [str(j) for j in list(range(int(t[0]), int(t[1])+1))]
elif "all" in args.resi or "All" in args.resi:
    resi = []
else:
    resi = args.resi.split(",")

if args.fixed is not None:
    if "-" in args.fixed:
        p = args.fixed.split("-")
        fix_res = [str(j) for j in list(range(int(p[0]), int(p[1])+1))]
    elif "," in args.fixed:
        fix_res = args.fixed.split(",")
else:
    fix_res = []

if "-" in args.pore:
    porelist = args.pore.split("-")
    poreinp = [str(i) for i in range(int(porelist[0]), int(porelist[1])+1)]
else:
    poreinp = args.pore.split(",")


with open('submit_MPNN_job.sh', 'w') as f:
    f.write(f'#!/bin/bash\n#SBATCH -p gpu\n#SBATCH --mem={args.mem}\n#SBATCH --gres=gpu:{args.gpu}:1\n#SBATCH -c 1\n#SBATCH -t 01:00:00\n#SBATCH --output=runMPNN.out\n\nsource activate mlfold\n\n')
    for pdbin in glob.glob(args.pdbs+"*.pdb"):
        poseinp = pose_from_file(pdbin)

        if args.cis_resid != '0':
            st, coords, pdbin = orient_pdb(pdbin, poseinp, args.cis_resid)
            poseinp = pose_from_file(pdbin)
        
        pdb = pdbin.split('/')[-1][:-4]
        inpfolder = "/".join(pdbin.split('/')[:-1])
        outfolder = args.outdir + pdb
        os.mkdir(outfolder)
        sh.copy(pdbin,outfolder)

        if args.symm:
            sym = poseinp.num_chains()
            Chid = core.pose.get_chain_id_from_chain('A', poseinp)
            pose = poseinp.split_by_chain(chain_id=Chid)
            if not resi:
                resi = [str(i) for i in range(1, pose.size()+1)]
        else:
            pose = poseinp
            if not resi:
                resi = [str(i) for i in range(1, pose.size()+1)]
        dssp = rosetta.core.scoring.dssp.Dssp(pose)
        dssp.dssp_reduced()
        dssp_str = dssp.get_dssp_secstruct()
        fixedres, intf_res, buried_res = None, None, None
        if args.interface:
            chain1 = ChainSelector("1")
            chain2 = ChainSelector("2")
            chainend = ChainSelector(f'{poseinp.num_chains()}')
            interface_selector1 = InterGroupInterfaceByVectorSelector( chain1, chain2 )
            interface_selector2 = InterGroupInterfaceByVectorSelector( chain1, chainend )
            interface_selector = OrResidueSelector(interface_selector1, interface_selector2)
            interface_chain1 = AndResidueSelector(chain1, interface_selector)
            intf_bool = interface_chain1.apply(poseinp)
            intf_bool = [x for j, x in enumerate(intf_bool) if j+1 <= len(dssp_str)]
            print(len(dssp_str),len(intf_bool))
            notintf = [str(i+1) for i, k in enumerate(dssp_str) if not intf_bool[i]]
            intfE_L = [str(i+1) for i, k in enumerate(dssp_str) if k not in ss and intf_bool[i]]
            fixedres = " ".join(notintf + intfE_L)
            chlist = "A B"
        elif args.inpaint:
            trb = pdb.split('_')
            data = np.load(inpfolder + '_'.join(trb[1:])[:-4]+".trb", allow_pickle=True)
            fixedres = " ".join([str(i+1) for i, x in enumerate(dssp_str) if data['mask_1d'][i]])
        elif len(resi) >= 1 and '0' not in resi:
            fixedres = " ".join([str(i+1) for i, k in enumerate(dssp_str) if str(i+1) not in resi])
            tiedres = []
            chlist = " ".join(chainstr[:pose.num_chains()])
        elif args.diffused:
            Chidiff = core.pose.get_chain_id_from_chain(args.diffused, pose)
            posediff = pose.split_by_chain(chain_id=Chidiff)
            dsspdiff = rosetta.core.scoring.dssp.Dssp(posediff)
            dsspdiff.dssp_reduced()
            dssp_strdiff = dsspdiff.get_dssp_secstruct()
            chlist = args.diffused
            seqnp = np.array([i for i, k in enumerate(posediff.sequence()) if k == 'G'])
            x = [0] + list(np.where(seqnp[1:] - seqnp[:-1] != 1)[0])
            GGlen = [x[i]-x[i-1] for i, k in enumerate(x) if i > 0]
            seqnp = [str(k+1) for i, k in enumerate(seqnp)]
            if GGlen[0] > 3:
                dres = [" ".join(seqnp[x[0]:x[1]+1])]
                dres = " ".join(dres + [" ".join(seqnp[x[i]+1:x[i+1]+1]) for i, k in enumerate(GGlen) if i > 0 and k > 3])
            else:
                dres = " ".join([" ".join(seqnp[x[i]+1:x[i+1]+1]) for i, k in enumerate(GGlen) if k > 3])
            fixedres = " ".join([str(i+1) for i, k in enumerate(dssp_strdiff) if str(i+1) not in dres.split(" ")])
            tiedres = []
        elif fix_res:
            fixedres = " ".join(fix_res)
            chlist = "A"
            tiedres = []
        if args.surface:
            intftmd, buriedtmd = find_intf_buried_res(st,args.TMdepth, args.interf_frac)
            pore_res, intf_res, buried_res = final_surf_res_for_design(poseinp, ss, intftmd, buriedtmd, dssp_str)
            buried_res = " ".join(buried_res)
            intf_res = " ".join(intf_res)

            if ss == "H":
                if args.pore:
                    fixedres = " ".join(poreinp)
                else:
                    fixedres = ""

        else:
            fixedres =" ".join([str(i+1) for i, k in enumerate(dssp_str) if k not in ss])
            if not fixedres.split():
                fixedres = []
            chlist = "A"
            tiedres = []
        if args.symm:
            tiedres1 = " ".join([str(i+1) for i, k in enumerate(dssp_str) if str(i+1) not in fixedres.split()])
            tiedres = ",".join([tiedres1 for num in range(sym)])
            if fixedres:
                fixedres = ",".join([fixedres for num in range(sym)])
            else:
                fixedres = []
            if intf_res and buried_res:
                intf_res = ",".join([intf_res for num in range(sym)])
                buried_res = ",".join([buried_res for num in range(sym)])
            chlist = " ".join([x for j, x in enumerate(chainstr) if j < sym])
        writeTask(f,fixedres,outfolder,args.nseq,args.bs,tiedres,chlist,buried_res,intf_res)

