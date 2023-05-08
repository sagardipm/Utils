#!/home/sagardip/.conda/envs/sdp-pyrosetta/bin/python

import sys
import glob
import os
import numpy as np
import argparse
from pyrosetta import *
from rosetta import *
from pyrosetta.rosetta.core.select.residue_selector import *
import shutil as sh
init()


chainstr = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pdbs', help="Path to pdbs whose secondary structure elements are to be designed with MPNN", action='store', type=str, required=True)

    parser.add_argument('--ss', help="Comma separated string of secondary structue elements to be designed, e.g E,L,H or E,L", action='store', type=str, default='L')
    
    parser.add_argument('--diffused', help="If true, all glycine positions with 3 or more consecutive glycines will be designed in the provided diffused chain id", action='store', type=str, default=None)

    parser.add_argument('--nseq', help="Number of MPNN sequences per input graft", action='store', type=int, default=32)

    parser.add_argument('--bs', help="Batch size for MPNN", action='store', type=int, default=8)
    
    parser.add_argument('--interface', help="Specify if you want to design chain1 interface. Can be coupled with secondary structure selection", action='store', type=bool, default=False)
    
    parser.add_argument('--symm', help="Bool for symetric pose. Must be true when MPNNing symmetric pose", action='store', type=bool, default=False)

    parser.add_argument('--inpaint', help="If inpaint true, residue indices will be selected from corresponding trb files", action='store', type=bool, default=False)
    
    parser.add_argument('--mem', help="Memory for gpu", action='store', type=str, default='16g')
    
    parser.add_argument('--resi', help="Residue string for design", action='store', type=str, default='0')
    
    parser.add_argument('--fixed', help="Residue indices to be kept fixed", action='store', type=str, default=None)

    parser.add_argument('--gpu', help="Type of gpu", action='store', type=str, default='a4000')
    
    parser.add_argument('--num_packs', help="Number of packed pdbs per sequence", action='store', type=str, default='1')
    
    parser.add_argument('--outdir', help="Output directory", action='store', type=str, default='./')
    
    parser.add_argument('--tmdesign', help="Specify if MPNN trained with global transmembrane label should be used", action='store', type=str, default="0")

    args = parser.parse_args()

    return args


def writeTask(file, fres, folder, Nseq, Nbatch, tiedres=[], ch="A"):
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
    elif args.tmdesign == "0":
        file.write(f'python /home/sagardip/MPNN_scPack/proteinmpnn/protein_mpnn_run.py --jsonl_path $path_for_parsed_chains --fixed_positions_jsonl $path_for_fixed_positions --chain_id_jsonl $path_for_assigned_chains --tied_positions_jsonl $path_for_tied_positions --out_folder $path_for_designed_sequences --num_seq_per_target {Nseq} --sampling_temp "0.1" --batch_size {Nbatch} --pack_side_chains 1 --num_packs {args.num_packs}\n\n\n')


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


with open('submit_MPNN_job.sh', 'w') as f:
    f.write(f'#!/bin/bash\n#SBATCH -p gpu\n#SBATCH --mem={args.mem}\n#SBATCH --gres=gpu:{args.gpu}:1\n#SBATCH -c 1\n#SBATCH -t 01:00:00\n#SBATCH --output=runMPNN.out\n\nsource activate mlfold\n\n')
    for pdbin in glob.glob(args.pdbs+"*.pdb"):
        pdb = pdbin.split('/')[-1][:-4]
        inpfolder = "/".join(pdbin.split('/')[:-1])
        outfolder = args.outdir + pdb
        os.mkdir(outfolder)
        sh.copy(pdbin,outfolder)
        poseinp = pose_from_file(pdbin)
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
        else:
            fixedres =" ".join([str(i+1) for i, k in enumerate(dssp_str) if k not in ss])
            if not fixedres.split():
                fixedres = ""
            chlist = "A"
            tiedres = []
        if args.symm:
            tiedres1 = " ".join([str(i+1) for i, k in enumerate(dssp_str) if str(i+1) not in fixedres.split()])
            tiedres = ",".join([tiedres1 for num in range(sym)])
            if fixedres:
                fixedres = ",".join([fixedres for num in range(sym)])
            else:
                fixedres = []
            chlist = " ".join([x for j, x in enumerate(chainstr) if j < sym])
        writeTask(f,fixedres,outfolder,args.nseq,args.bs,tiedres,chlist)

