"""
Author: Doruk Efe Gökmen
Date: 30/11/2020

Implementation of pocket Monte Carlo algorithm of Krauth et al.
"""

module Pocket

export pocket_step!, sites_bonds

using StatsBase
using Random

include("RandomDimers.jl")
include("DimerMove.jl")
include("Lattice.jl")
include("DrawDimerConfigs.jl")


function delete_ntimes(a, item, n::Int)
	"""
	Function to delete the item in an array with certain value only n times.
	The item should be a scalar.
	"""
    a_new = []
    count = 0
    for el in a
        if el == item && count < n
            count += 1
        else
            append!(a_new, el)
        end
    end
    return a_new
end


function cart_id(id::Int, n::Int)
	"""
	Constructs index in Cartesian coordinates as a dim-tuple
	from the linear index id.
	"""
    return [div(id-1, n^(dim-j))%n for j=[1:dim;]] + [1 for d=[1:dim;]]
end

function lin_id(cart_id, n::Int)
	"""
	Constructs the linear index as a single scalar.
	"""
     return sum(n^(dim-j)*(cart_id[j]-1) for j in [1:dim;]) + 1
end


function reflect_dimer!(A::Array{Neighbour{Int8},2}, id::Int, nn::Int, n::Int)
	"""
	Modifies the dimer configuration by reflecting the dimer (id, nn)
	across a symmetry axis (currently specific to the diagonal axis).
	"""
    id_nn = A[id, nn+1].site
    A[id, nn+1].J -= Int8(1)
    A[id_nn, invert(nn)+1].J -= Int8(1)
    #set_dimer!(A, id, nn, Int8(0))
    id_c = cart_id(id, n)
    id_c[1:2] = [id_c[2], id_c[1]]
    
    id_new = lin_id(id_c, n)
    nn_new = (nn+2)%(2*dim)
    
    id_nn_new = A[id_new, nn_new+1].site
    A[id_new, nn_new+1].J += Int8(1)
    A[id_nn_new, invert(nn_new)+1].J += Int8(1)
    #set_dimer!(A, id_new, nn_new, Int8(1))
    return id_new, nn_new
end


function pocket_step!(A::Array{Neighbour{Int8},2}, n::Int)
	"""
	Single step of the pocket algorithm.
	"""
	N = n^dim 

	k = 1
	nns = []
	while length(nns) == 0
	    k = rand(1:N) # randomly select a site
	    nns = get_dimer(A, k)
	end

	k_nn = A[k, nns[1]+1].site
	pocket = [(k, nns[1]), (k_nn, invert(nns[1]))]
	avail = [i for i=setdiff([1:N;], pocket) if length(get_dimer(A, i)) > 0];

	while length(pocket) > 0
	    (i, nn) = pocket[1]
	    i_nn = A[i, nn+1].site
	    i_ref, nn_ref = reflect_dimer!(A, i, nn, n)
	    i_nn_ref = A[i_ref, nn_ref+1].site
	    for j in avail
	        dimers = get_dimer(A, j)
	        if j == i_ref && length(get_dimer(A, i_ref)) > 1 
	            nn_j_old = delete_ntimes(get_dimer(A, i_ref), nn_ref, 1)[1]
	            append!(pocket, [(j, nn_j_old), (A[j, nn_j_old+1].site, invert(nn_j_old))])
	        elseif j == i_nn_ref && length(get_dimer(A, i_nn_ref)) > 1 
	            nn_j_old = delete_ntimes(get_dimer(A, i_nn_ref), invert(nn_ref), 1)[1]
	            append!(pocket, [(j, nn_j_old), (A[j, nn_j_old+1].site, invert(nn_j_old))])
	        end
	    end
	    setdiff!(pocket, [(i, nn), (i_nn, invert(nn))]) 
	end

end


function sites_bonds(A::Array{Neighbour{Int8},2}, n::Int, site_corr::Bool)
    """
    Converts the dimer configuration into a matrix containing binary dimer variables
    and (spurious) noisy site variables (correlated or uncorrelated with faces).
    """
    N = n^dim # system size

    X = zeros(2n, 2n) # initialise sites_bonds matrix

    for i in 0:N-1
        k = div(i,n)
        l = i % n 

        X[2k+1, 2l+1] = rand(0:1) # place noise on sites

        if site_corr # fictitious sites at plaquette faces
            X[2k+2, 2l+2] = X[2k+1, 2l+1]
        else
            X[2k+1, 2l+1] = rand(0:1)
        end
        
        # place dimers: 
        if A[i+1, 1].J == 1 && 2l+2 < 2n+1 
            X[2k+1, 2l+2] = 1 # dimer to right
        elseif A[i+1, 2].J == 1 && 2l > 0
            X[2k+1, 2l] = 1 # dimer to left
        elseif A[i+1, 3].J == 1 && 2k+2 < 2n + 1
            X[2k+2, 2l+1] = 1 # dimer to bottom
        elseif A[i+1, 4].J == 1 && 2k > 0
            X[2k, 2l+1] = 1 # dimer to top
        end
    end
    return X
end

end # ends the module
