"""
Author: Maciej Koch-Janusz
Date: 13/06/2022
"""

import math
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from collections import Counter


def eids_from_edges(reference_edges_list,edges_by_vertices):
    """
    Return the list unique edge IDs corresponding to the edges (given as pairs of vertices) appearing in the second argument.
    """
    aux1=[(reference_edges_list == i).all(axis=1).nonzero()[0] for i in np.vstack((edges_by_vertices,np.fliplr(edges_by_vertices)))]
    return sorted([a[0] for a in aux1 if a.size != 0])
    

def non_collapsed_estimates(x, epsilon = 0.05):
    """
    Filter for values of MI in input x, larger than a specified cutoff.
    """
    aux_x = [x >= epsilon]
    return aux_x


def non_collapsed_data(x, non_collapsed, ignore_collapse = False, return_item = -2):
    """
    By default returns the last non-collapsed -- according to MI estimates -- piece of data (filter),
    or the specific return_item element (to be used to take e.g. early training results).
    """
    if not(ignore_collapse):
        if len(np.where(np.isnan(x[tuple(non_collapsed)][return_item]))[0])!= 0:
            raise ValueError("FILTERS COLLAPSED: NaNs !!!")
        return x[tuple(non_collapsed)][return_item] # the filter collapses 1 iteration BEFORE the information drops!
    else:
        return x[-1]


def non_collapsed_estimates_ewm(est, non_collapsed,ewm_span=50, ignore_collapse = False, return_item = -2):
    """
    Return the Exponential Moving Average of the noncollapsed part of the input est.
    """
    if ignore_collapse:
        return non_collapsed_data(est,non_collapsed, ignore_collapse=ignore_collapse)
    else:
        if ewm_span == 0:
            return non_collapsed_data(est,non_collapsed,return_item = return_item)
        else:
            nonc_est = est[tuple(non_collapsed)]
            return pd.Series(nonc_est).ewm(span=ewm_span).mean().iloc[return_item]

def load_Vs(samples, case_list, vind):
    """
    Load the V samples corresponding to the selected seeds, vertex and case.
    samples -- list of sample file IDs (seeds)
    case_list -- list of cases which define the shape of V
    vind -- the index of vertex around which Vs are taken
    """
    sample_no = int("".join([str(i) for i in samples]))
    print(samples, sample_no)
    
    Vs_dict = {}
    for load_case in case_list:
        # Vs are organized in files per sample batch (seed). If using multiple sample batches, concatenate and reshape.
        Vs_list = []
        for sample_no in samples:
            Vs_list.append(np.load(r"/home/cluster/mkochj/scratch/sobisw/data/EandV/Vs_%i_vi%i_c%i.npy" % (sample_no,vind,load_case)))
        Vs_dict[load_case] =  np.reshape(Vs_list,newshape=(len(Vs_list)*Vs_list[0].shape[0],Vs_list[0].shape[1],1)) # REMEMBER about the additional axis for Vs !!!!!
        print("Case: %i, shape Vs: "%(load_case), Vs_dict[load_case].shape)
    return Vs_dict
       
            
            
def load_filters_estimates_dict(data_dir, case_list, num_of_runs=None, run_list=None, cGV_edges=None, EV_params=None, CG_params = None, disc=1, ignore_collapse = False, return_item = None):
    """
    Populate the dictionaries with (final, non-collapsed) pre-trained filters and MI estimates.
    Dictionary keys given by case_list.
    data_dir -- base directory holding trained filters
    case_list -- list of CG paramater "cases" to be loaded
    num_of_runs -- num of independent optimization runs loaded for each case
    cGV_edges -- dictionary of edges of the region V for each case
    EV_params, CG_params -- general params/hyperparams of the CG/optimization
    disc -- 
    ignore_collapse -- False if we take care to find non_collapsed MI estimates/filters (as opposed to simply taking the final ones)
    return_item -- manually specify the filter/MI num to be returned (from the non_collapsed). E.g. to check early training.
    """
    c_filters_runs = {}
    c_estimates_runs = {}
    
    if run_list is None:
        run_list = list(range(1,num_of_runs+1))
    else:
        num_of_runs = len(run_list)

    
    if not(return_item):
        return_item = {key: -2 for key in case_list}
    
    for case_no in case_list:
        aux_filters_runs = np.zeros(shape=(num_of_runs,len(cGV_edges[case_no]),CG_params['num_hiddens']))
        aux_estimates_runs = np.zeros(shape=(num_of_runs,1))
        run_no=1 # while construction is better for variable size list (if files missing)
        extant_run_no = 1
        #while run_no <= num_of_runs:
        #    try:
        #        aux_filters = np.load(data_dir + 'filters_%i_vi%i_c%i_h%i_disc%i_run%i.npy'%(EV_params['sample_no'],EV_params['V_index'],case_no,CG_params['num_hiddens'],disc,run_no))
        #        aux_estimates = np.load(data_dir + 'estimates_%i_vi%i_c%i_h%i_disc%i_run%i.npy'%(EV_params['sample_no'],EV_params['V_index'],case_no,CG_params['num_hiddens'],disc,run_no))
        #        non_collapsed = non_collapsed_estimates(aux_estimates)
        #        aux_filters_runs[extant_run_no-1] = non_collapsed_data(aux_filters,non_collapsed,ignore_collapse=ignore_collapse,return_item = return_item[case_no])
        #        aux_estimates_runs[extant_run_no-1] = non_collapsed_estimates_ewm(aux_estimates,non_collapsed,ewm_span=100,ignore_collapse=ignore_collapse,return_item = return_item[case_no])
        #        run_no +=1
        #        extant_run_no +=1
        #    except FileNotFoundError:
        #        print("FILE ERROR: Case,run: ", case_no,run_no," not found. Ignoring.")
        #        run_no +=1 # but we *do not* increase the extant_run_no index.
                
        for run_no in run_list:
            try:
                aux_filters = np.load(data_dir + 'filters_%i_vi%i_c%i_h%i_disc%i_run%i.npy'%(EV_params['sample_no'],EV_params['V_index'],case_no,CG_params['num_hiddens'],disc,run_no))
                aux_estimates = np.load(data_dir + 'estimates_%i_vi%i_c%i_h%i_disc%i_run%i.npy'%(EV_params['sample_no'],EV_params['V_index'],case_no,CG_params['num_hiddens'],disc,run_no))
                non_collapsed = non_collapsed_estimates(aux_estimates)
                aux_filters_runs[extant_run_no-1] = non_collapsed_data(aux_filters,non_collapsed,ignore_collapse=ignore_collapse,return_item = return_item[case_no])
                aux_estimates_runs[extant_run_no-1] = non_collapsed_estimates_ewm(aux_estimates,non_collapsed,ewm_span=100,ignore_collapse=ignore_collapse,return_item = return_item[case_no])
                extant_run_no +=1
            except FileNotFoundError:
                print("FILE ERROR: Case,run: ", case_no, run_no," not found. Ignoring.")

        c_filters_runs[case_no] = aux_filters_runs[0:extant_run_no-1]   #Remove the trailing zeros (from initialization)
        c_estimates_runs[case_no] = aux_estimates_runs[0:extant_run_no-1]
        
    return c_filters_runs, c_estimates_runs
    
    
def single_filter_visualize(V_edgelist, filters, filter_no, hidden_no=0, edges=None, nodes=None, nodepos=None, node_size=1, width=3):
    """
    Plots the single filter on the ngraph of V. Doesn't draw the full graph.
    V_edgelist -- list of edge IDs in V
    filters -- the series of trained filters
    filter_no -- index of filter in the filters series to be plotted
    hidden_no -- which filter component to plot (if the filter is multicomponent)
    """
    # ----- Prepare data, graph, .... --------------
    V_edgelist = V_edgelist

    num_edges=np.shape(edges)[0]
    V_edges = np.zeros(num_edges)
    V_edges[V_edgelist]=1

    #filter_densities = np.zeros((num_edges,CG_params['num_hiddens']))
    filter_densities = np.zeros(num_edges)
    filter_densities[V_edgelist] = filters[filter_no][:,hidden_no]
    #filter_densities[V_edgelist] = filters[filter_no][:,:]

    R=nx.Graph()
    elist = [(edges[i,0],edges[i,1],{'colors': filter_densities[i]}) for i in range(num_edges) if V_edges[i]==1]
    R.add_edges_from(elist)

    num_vtx=np.shape(nodes)[0]
    ndict = {nodes[i]:{'pos':(nodepos[i,0],nodepos[i,1])} for i in range(num_vtx)}
    nx.set_node_attributes(R, ndict)

    ecolors=[R[u][v]['colors'] for u,v in R.edges()]
    pos_V=nx.get_node_attributes(R,'pos')
    
    # ----- Plot ---------------
    
    fig = plt.figure(1)
    #ax1 = plt.subplot(1,1,1)
    plt.gca().set_aspect('equal')
    nx.draw(R,pos_V,node_color='black',node_size=node_size,edge_color=np.array(ecolors)[:],edge_cmap=plt.get_cmap('bwr'),width=width)
    #nx.draw(R,pos_V,node_color='black',node_size=node_size,edge_color=np.array(ecolors)[:,hidden_no],edge_cmap=plt.get_cmap('bwr'),width=width)
    
    
def small_V_visualize(V_edgelist, edge_densities, edges=None, nodes=None, nodepos=None, n_rows =1, n_cols=1, title="", figsize=(10.4,6.8), node_size=1, width=3, layout_dict = {"pad":0.0, "h_pad": 0.08, "w_pad": -10.08, "rect":(0,0,1,1)}):
    """
    Plot multicomponent data (filters,PCA components,....) on the V subgraph  (doesn't draw the full system).
    Ensure that number of subplots is equal to the number of data components.
    V_edgelist -- list of edge IDs in V
    edge_densities -- mutlicomponent data to be plotted on V
    """
    # ----- Prepare data, graph, .... --------------
    R=nx.Graph()
    elist = [(edges[e_ind,0],edges[e_ind,1],{'colors': edge_densities[num,:]}) for num,e_ind in enumerate(V_edgelist)]
    R.add_edges_from(elist)
    
    num_vtx=np.shape(nodes)[0]
    ndict = {nodes[i]:{'pos':(nodepos[i,0],nodepos[i,1])} for i in range(num_vtx)}
    nx.set_node_attributes(R, ndict)

    ecolors=[R[u][v]['colors'] for u,v in R.edges()]
    pos_V=nx.get_node_attributes(R,'pos')
    
    # ----- Plot subfigures ---------------
    layout_dict = layout_dict
    n_rows=n_rows
    n_cols=n_cols
    if n_rows*n_cols != edge_densities.shape[-1]:
        print("WARNING: num of sublots and components do not match: ",n_rows*n_cols,edge_densities.shape[-1])
    
    fig = plt.figure(tight_layout=layout_dict, figsize=figsize) #figure(1,)
    axes_plots = []

    for plot_ind in range(1,n_rows*n_cols+1):
        try:
            axes_plots.append(plt.subplot(n_rows,n_cols,plot_ind))
            plt.gca().set_aspect('equal')
            plt.gca().set_title(title+str(plot_ind))
            nx.draw(R,pos_V,node_color='black',node_size=node_size,edge_color=np.array(ecolors)[:,plot_ind-1],edge_cmap=plt.get_cmap('bwr'),width=width)
        except IndexError:
            print("No data components for subplot", plot_ind)


def full_graph_visualize(vis_edgelist, edge_densities=None, edges=None, nodes=None, nodepos=None, n_rows=1, n_cols=1, title="", figsize=(10.4,6.8), node_size=1, width=3, layout_dict = {"pad":0.0, "h_pad": 0.08, "w_pad": -10.08, "rect":(0,0,1,1)}):
    """
    Plot multicomponent data (filters,PCA components,....) on the full system graph.
    Ensure that number of subplots is equal to number of data components.
    If edge_densities is None, will draw indicator function on vis_edglist.
    vis_edgelist -- list of edge IDs on which data will be drawn
    edge_densities -- multicomponent data to be plotted on the edges in vis_edgelist    

    Example uses:
    full_graph_visualize(cGV_edges[load_case], edges=edges, nodes=nodes, nodepos=nodepos)
    full_graph_visualize(cGV_edges[load_case], filters[-1][:,0],edges=edges, nodes=nodes, nodepos=nodepos)
    full_graph_visualize(eighthoods, edges=edges, nodes=nodes, nodepos=nodepos)
    """
    
    #ddensity = np.zeros(872)
    #ddensity[V_edgelist] = filters[-1][:,0]
    
    visualized_edges = np.zeros(len(edges))
    if edge_densities is None:   # you *have to use* "is" and not ==, the latter for arrays returns bool array!!!
        visualized_edges[vis_edgelist]=1
    else:
        visualized_edges[vis_edgelist]=edge_densities[:]

    R=nx.Graph()
    
    num_vtx=np.shape(nodes)[0]
    for i in range(num_vtx) :
        R.add_node(nodes[i],pos=(nodepos[i,0],nodepos[i,1]))
    print ("number of nodes/vertices:",np.shape(nodes))

    num_edges=np.shape(edges)[0]
    for i in range(num_edges):
        #R.add_edge(edges[i,0],edges[i,1], width=VE_edges[i],color=ddensity[i])
        R.add_edge(edges[i,0],edges[i,1],color=visualized_edges[i])
    pos=nx.get_node_attributes(R,'pos')
    ecolors=[R[u][v]['color'] for u,v in R.edges()]
    #ewidths=[R[u][v]['width'] for u,v in R.edges()]

    plt.figure(figsize=(58,58))
    plt.gca().set_aspect('equal')
    nx.draw(R,pos,node_color='black',node_size=10,edge_color=ecolors,edge_cmap=plt.get_cmap('bwr'),width=4)
    plt.show()
    #plt.close()
            
            
def filter_PCA(case_list,filters,n_components = 4):
    """
    Returns a dictionary (keys = cases) of fitten PCA objects (components_, singular_values_,...) 
    *** Currently assuming a single hidden !!!
    """
    pca_dict = {}
    for case_no in case_list:
        pca_dict[case_no] = PCA(n_components=n_components) #,svd_solver='full'
        pca_dict[case_no].fit_transform(filters[case_no][:,:,0])
        #print(pca_dict[case_no].components_.shape)
    return pca_dict
    
    
def hexbin_visualize(data1,data2,xlims=None,ylims=None,title="",gridsize=25,aspect=1., cmap = 'Reds'):
    """
    Plot a 2D histogram with hexagonal bins.
    """
    if xlims == None:
        xlims = 1.1*max(abs(data1))
    if ylims == None:
        ylims = 1.1*max(abs(data2))
        
    plt.figure()   # needs this to create a new figure or it will plot on the previous one!
    plt.hexbin(data1, data2, gridsize=gridsize, cmap=cmap)
    plt.xlim([-xlims,xlims])
    plt.ylim([-ylims,ylims])
    plt.gca().set_title(title)
    plt.gca().set_aspect(aspect)
    
    
def filterPCA_hexbin_visualize(filters,pca,c1,c2,xlims=None,ylims=None,title="",aspect=1.,cmap = 'Blues'):
    """
    2D hexbin of projections of filters onto 2 selected PCA components
    ***** Currently assumes a single hidden for filters !!!
    """
    projections= np.matmul(filters[:,:,0],pca.components_.transpose())
    
    np.set_printoptions(precision=3)
    print(pca.singular_values_/max(pca.singular_values_), "Runs: ", filters.shape[0])
    
    hexbin_visualize(projections[:,c1],projections[:,c2],xlims=xlims,ylims=ylims,title=title+"Cx = %i, Cy=%i"%(c1,c2),gridsize=25,aspect=aspect,cmap=cmap)
    
    
def dot_prod_V(configs,vis_edgelist,edge_densities):
    """
    Apply the filter (PCA comp,...) given by edge densitities to configuration(s) on edges in vis_edgelist
    !!!! edge_densities assumed of shape (num_edges,num_components) !!!!
    !!!! configs of shape (num_configs,  num_edges) !!! for a single one reshape to (1, num edges)
    !!!! remember Vs have this additional axis, reshape before feeding as an argument.
    """
    
    # if configs are actually Vs then the ordering is already correct (but remember about the additional axis for V)
    if configs.shape[1] == edge_densities.shape[0]:
        dot_prod = np.matmul(configs,edge_densities)
        
    # otherwise, select the appropriate edges from Xs (which don't have the final additional axis):
    else:
        dot_prod = np.matmul(configs[:,vis_edgelist],edge_densities)
            
    return dot_prod
    

def match_edges(original_edges,target_edges,original_vertex,target_vertex,nodes,nodepos,edges):
    """
    find indices of edges of original in the target list. 
    Note that the central vertex V389 is at [0,0].
    """
    # 1. translate edges to pairs of nodes
    # 2. write each node using its 2D absolute coordinates
    # 3. For each edge, sort its coords lexicographically, so edges are easy to compare
    # 4. For each edge in the original scan target and find the index of matching edge 
    tt_coords = np.array([[nodepos[int(edges[edge][0])],nodepos[int(edges[edge][1])]] for edge in target_edges])
    tt_coords_lexsort_inside_edge = [edge[np.lexsort(np.rot90(edge))] for edge in tt_coords]

    trsl_coords = np.array([[nodepos[int(edges[edge][0])] +nodepos[target_vertex]-nodepos[original_vertex], nodepos[int(edges[edge][1])]+nodepos[target_vertex]-nodepos[original_vertex]] for edge in original_edges])
    trsl_coords_lexsort_inside_edge = [edge[np.lexsort(np.rot90(edge))] for edge in trsl_coords]


    subarr_pos=[]
    for j,subarr in enumerate(trsl_coords_lexsort_inside_edge):
        #print(j,subarr)
        for i,subarr2 in enumerate(tt_coords_lexsort_inside_edge):
            #if np.all(subarr == subarr2): #exact comparison fails due to numerical errors in position diffs
            if np.all(np.isclose(subarr,subarr2)):
                subarr_pos+=[i]
                
    return subarr_pos
    
    
def corr2_coeff(A, B):
    """
    Return the linear Pearson correlation coefficient (matrix) of the input arrays.
    Inputs are assumed to be of shape: (Num_features_A/B, Num_samples), and the output (Num_features_A,Num_features_B)
    """
    
    A_mA = A - A.mean(axis=1)[:, None]
    B_mB = B - B.mean(axis=1)[:, None]

    ssA = (A_mA**2).sum(axis=1)
    ssB = (B_mB**2).sum(axis=1)

    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    
    
def k_vertices(edges,k):
    """
    Returns the list of vertices of valence k
    edges -- list of graph edges as pairs of vertices
    k -- (int) valence (= num of incident edges) of vertices sought
    """
    edgecount_0 = Counter(edges[:,0])
    edgecount_1 = Counter(edges[:,1])
    
    kfold_0 = {key for key in edgecount_0 if edgecount_0[key]==k}
    kfold_1 = {key for key in edgecount_1 if edgecount_1[key]==k}
    kfold = list(kfold_0 | kfold_1)
    kfold.sort()
    return kfold
    

def pd_add_zeros(frame,bins):
    """
    Add the "missing" columns/rows in the pd dataframe with zero entries
    frame -- pandas dataframe, potentially missing cols/rows for empty bins
    bins -- list of desired bins
    """
    missing_cols = list(set(range(1,bins+1)) - set(frame.columns))
    frame[missing_cols] = 0
    frame = frame.reindex(sorted(frame.columns), axis=1)
    
    missing_rows = list(set(range(1,bins+1)) - set(frame.index))
    for mr in missing_rows:
        frame.loc[mr] = [0]*bins
    frame.sort_index(inplace=True)
    
    return frame 