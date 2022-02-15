"""
Execution interface for generating an ensemble of
RSMI optimisation instances.

Author: Doruk Efe Gökmen
Date: 04/08/2020
"""

import os
import json
import argparse
import pickle
import numpy as np

import rsmine.coarsegrainer.build_dataset as ds
import rsmine.coarsegrainer.cg_optimisers as cg_opt
import rsmine.coarsegrainer.plotter


def coarsegrain(index, ll, buffer_size=2, env_cap=4, ensemble_size=1,
                plot=False, filter_lim=0.2, use_GPU=False, 
                Dir='~/RSMI-NE/coarsegrainer/data/results'):

    V, e = generator.rsmi_data(index, ll, buffer_size=buffer_size, cap=env_cap)
    V = np.reshape(V, (np.shape(V)[0],) +  ll + (1,) )
    V, e = ds.array2tensor(V), ds.array2tensor(e)

    estimates = {}
    coarse_vars = {}
    filters = {}
    Λ_net = {}

    for ens_id in range(ensemble_size):
        print('Realisation ', ens_id+1)
        for estimator, mi_params in estimators.items():
            print("Training %s..." % estimator)
            estimates[ens_id], coarse_vars[ens_id],\
            filters_all, Λ_net[ens_id] \
                = cg_opt.train_RSMI_optimiser(e, V, 
                                    CG_params, critic_params, opt_params,\
                                    mi_params, data_params, plot=plot, 
                                    filter_lim=filter_lim, use_GPU=use_GPU)
        filters[ens_id] = filters_all[-1]

    mi_filename = ds.filename(data_params['model'], data_params['lattice_type'], 
                              L=data_params['L'], T=data_params['T'], 
                              J=data_params['J'], fileformat='pkl', basedir=Dir, 
                              srn_correlation=data_params['srn_correlation'],
                              prefix=('miestimates_buffer%i' %buffer_size))

    filter_filename = ds.filename(data_params['model'], data_params['lattice_type'], 
                                  L=data_params['L'], T=data_params['T'], 
                                  J=data_params['J'], fileformat='pkl', basedir=Dir,
                                  srn_correlation=data_params['srn_correlation'], 
                                  prefix=('filters_buffer%i' %buffer_size))

    # save the outputs
    with open(mi_filename, 'wb') as f:
        pickle.dump(estimates, f)
    with open(mi_filename, 'rb') as f:
        estimates = pickle.load(f)

    with open(filter_filename, 'wb') as f:
        pickle.dump(filters, f)
    with open(filter_filename, 'rb') as f:
        filters = pickle.load(f)

    Λ_net['InfoNCE'].save_weights('./checkpoints/lambda_net')

    for k in estimators.keys():
        plotter.plot_fancy_rsmimax(estimates, filters, opt_params, CG_params,
                                   generator, N_samples=data_params['N_samples'],
                                   mi_bound=k, series_skip=data_params['N_samples']\
                            //(opt_params['batch_size']*3)*opt_params['iterations'],
                                   filter_lim=filter_lim, save=True,
                                   interpolation='nearest', cmap='RdBu')

    return estimates, coarse_vars, filters, Λ_net


class RawDescriptionArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, 
                                                    argparse.RawDescriptionHelpFormatter):
	pass

if __name__ == "__main__":
	#Command line argument parser
    parser = argparse.ArgumentParser(description='Finds the optimal coarse-graining filter.', 
                                    formatter_class=RawDescriptionArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--inputprefix', type=str, default='params', 
                        help='prefix name for the input file for rsmi calculation.')
    parser.add_argument('-x', '--xindex', type=int, default=9, 
                        help='x index of visible block')
    parser.add_argument('-y', '--yindex', type=int, default=9, 
                        help='y index of visible block')
    parser.add_argument('-b', '--buffersize', type=int, default=2, 
                        help='width of the buffer')
    parser.add_argument('-E', '--envsize', type=int, default=4,
                        help='width of the environment')
    parser.add_argument('-e', '--ensemblesize', type=int, default=1, 
                        help='number of instances for RSMI optimisation')
    parser.add_argument('-a', '--filterlim', type=float, default=0.2, 
                        help='density plot color scale limit for the filters')
    parser.add_argument('-g', '--useGPU', default=None, action='store_true', 
                        help='use a GPU node for RSMI optimisation')
    parser.add_argument('-r', '--resultdir', type=str, default=None, 
                        help='enter directory for output results')
    parser.add_argument('-B', '--basedir', type=str, default=None, 
                        help='enter full directory for input data')
    args = parser.parse_args()

    prefix = args.inputprefix
    result_dir = args.resultdir
    base_dir = args.basedir
    index = (args.xindex, args.yindex)
    buffer_size = args.buffersize
    env_size = args.envsize
    ensemble_size = args.ensemblesize
    filter_lim = args.filterlim
    use_GPU = args.useGPU

    with open(os.path.join('input', prefix+'.json')) as f:
        params = json.load(f)

    data_params = params['data_params']
    CG_params = params['cg_params']
    critic_params = params['critic_params']
    opt_params = params['opt_params']
    estimators = params['estimators']

    CG_params['ll'] = tuple(CG_params['ll'])
    ll = CG_params['ll']

    generator = ds.dataset(**data_params, basedir=base_dir)

    if len(generator.configurations) >= data_params["N_samples"]:
        generator.configurations = generator.configurations[0:data_params["N_samples"]]
        generator.N_samples = generator.N_configs = len(generator.configurations)
    else:
        print('Sample dataset is smaller than the specified number of configurations.')
        generator.N_samples = generator.N_configs = len(generator.configurations)

    estimates, coarse_vars, filters, Λ_net = coarsegrain(index, ll, 
                                                        buffer_size=buffer_size, 
                                                        env_cap=ll[0]+2*(buffer_size+env_size),
                                                        ensemble_size=ensemble_size, 
                                                        filter_lim=filter_lim, 
                                                        use_GPU=use_GPU, Dir=result_dir)
