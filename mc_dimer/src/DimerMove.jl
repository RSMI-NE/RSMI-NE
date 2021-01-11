"""
Author: Doruk Efe Gökmen
Date: 07/04/2020

Define methods that modify the dimers on the lattice structure.
Also evaluate certain properties of the dimer configuration.
"""

export invert
export set_dimer!, get_dimer
export parallel_count
export compute_weights
export compute_parallel_counts

include("Lattice.jl")

function invert(k::Int)
    """
    Inverts neighbour index:
    r(0),l(1),b(2),u(3) -> l,r,u,b
    """
    return k + (-1)^k
end


function set_dimer!(A::Array{Neighbour{Int8},2}, i::Int, nn::Int, dimer::Int8)
    """
    Sets the state of the link between site i 
    and its nn'th neighbour.
    dimer=1: dimer
    dimer=0: no dimer
    """
    i_nn = A[i, nn+1].site # get neighbour index
    A[i, nn+1].J = dimer # place or remove dimer
    A[i_nn, invert(nn)+1].J = dimer
end


function get_dimer(A::Array{Neighbour{Int8},2}, i::Int)
    """
    Returns the list of neighbours of i
    that are connected to it by a dimer.
    Note: defects are allowed.
    """
    dimers = []
    
    for nn in 0:D-1
        if A[i, nn+1].J > 0
            for r in 1:A[i, nn+1].J 
                append!(dimers, nn)
            end
        end
    end
    return dimers
end


function parallel_count(A::Array{Neighbour{Int8},2}, i::Int, nn::Int)
    """
    Counts the number of parallel dimers 
    to the one connected to site i
    """
    h_set = [0, 1]
    v_set = [2, 3]
    
    N_parallels = 0
    
    if nn in h_set # count = terms
        for v in v_set
            i_nn = A[i, v+1].site
            N_parallels += A[i_nn, nn+1].J
        end
    elseif nn in v_set # count || terms
        for h in h_set
            i_nn = A[i, h+1].site
            N_parallels += A[i_nn, nn+1].J
        end
    end
        
    return N_parallels
end

function compute_parallel_counts(A::Array{Neighbour{Int8},2}, j::Int)
    counts = Array{Int64,1}(UndefInitializer(), D)

    for nn_j in 0:D-1
        counts[nn_j+1] = parallel_count(A, j, nn_j)
    end
    return counts
end


function weight(A::Array{Neighbour{Int8},2}, i::Int, nn::Int, T::Float64, v::Int=-1)
    """
    Unnormalised Boltzmann weight corresponding to the dimer
    model with aligning (v<0) (anti-aligning (v>0)) interactions
    """
    return exp(-v*parallel_count(A, i, nn)/T)
end


function compute_weights(A::Array{Neighbour{Int8},2}, j::Int, T::Float64, v::Int=-1)
    """
    Compute Boltzmann weights for all D possible dimer configurations a t site j
    """
    weights = Array{Float64,1}(UndefInitializer(), D)
    
    for nn_j in 0:D-1
        weights[nn_j+1] = weight(A, j, nn_j, T, v)
    end
    return weights
end