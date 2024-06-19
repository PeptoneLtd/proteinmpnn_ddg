import argparse
from proteinmpnn_ddg import predict_logits_for_all_point_mutations_of_single_pdb

if (__name__=='__main__'):
    parser = argparse.ArgumentParser(description='Uses ProteinMPNN-ddG to compute fitness of all point mutations of a pdb for a particular chain')
    parser.add_argument("--pdb_path", help="", required=True)
    parser.add_argument("--chains", default='A', help='Chains to load from PDB as prediction context, separated by commas e.g. `A,B`')
    parser.add_argument("--chain_to_predict", default=None, help='Chain to predict mutations of, defaults to the first chain in --chains if not specified')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--nrepeats", default=1, type=int, help='Runs model multiple times with different seeds (split from input seed) for averaged prediction')
    parser.add_argument("--outpath", required=True, help='CSV path to write outputs to')
    parser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    parser.add_argument("--without_ddG_correction", action="store_true", help="Whether to not apply correction for ddG")
    args = parser.parse_args()
    if (not args.without_ddG_correction):
        assert (args.model_name=='v_48_020'), 'ddG correction only appropriate with v_48_020 model, please rerun with `--without_ddG_correction` flag'
    df = predict_logits_for_all_point_mutations_of_single_pdb(args.model_name, args.chains.split(','), args.pdb_path, nrepeat=args.nrepeats, seed=args.seed, chain_to_predict=args.chain_to_predict, pad_inputs=False, apply_ddG_correction=not args.without_ddG_correction)
    df.to_csv(args.outpath)
