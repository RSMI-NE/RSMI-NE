"""
Author: Doruk Efe Gökmen
Date: 25/04/2020

Construct data structure that stores the variable lattice connectivity.
Convert between derived data structure and some compressed representation 
(states matrix) of the lattice connectivity.
"""


export Neighbour
export dim, D


dim = 2 # dimensionality of system
D = 2*dim # number of nearest neighbours in hypercubic lattice

 
mutable struct Neighbour{T}
    """
    custom type to express variable lattice structure
    """
    site::Int
    J::T
end


function set_lattice(n::Int)
    """
    Initiates a hypercubic lattice graph in D-dimensions
    Each vertex is a list of composite type Neighbour 
    containing site index and unspecified dimer links 
    """
    N = n^dim # system size

    vertices = Array{Neighbour{Int8},2}(UndefInitializer(), N, D)
    
    for i in 0:N-1
        j = i # partial index
        
        nn = 1
        for d in 0:dim-1
            a = j % n # component for current dimension
            
            for p in 0:1
                if p % 2 == 1
                    site = i + ((a - 1 + n) % n - a) * n^d + 1
                else
                    site = i + ((a + 1) % n - a) * n^d + 1
                end  
                vertices[i+1, nn] = Neighbour{Int8}(site, 0)
                nn += 1
            end
            j = div(j,n) # advance to next dimension via integer division
        end
    end
    return vertices
end       


function set_state!(A::Array{Neighbour{Int8},2}, M::Array{Int8,2}, n::Int)
    """
    Sets up the states in the hypercubic lattice according to array M
    """
    N = n^dim # system size

    for i in 0:N-1
        state = M[div(i,n)+1, i%n+1]
        
        for nn in 0:D-1
            if nn == state
                A[i+1, nn+1].J = 1 # add dimer according to state matrix
            else
                A[i+1, nn+1].J = 0
            end
        end
    end
end
    

function get_state(A::Array{Neighbour{Int8},2}, n::Int)
    """
    Constructs the states matrix M from the lattice A of Neigbour type objects
    """
    N = n^dim # system size

    M = zeros(Int8, n, n) # initialise state matrix
    
    for i in 0:N-1
        for nn in 0:D-1
            if A[i+1, nn+1].J == 1
                M[div(i,n)+1, i%n+1] = nn
            end
        end
    end
    return M
end

function delta2state(delta::Array{Int,1})
    if delta == [0, -1]
        state = 0
    elseif delta == [0, 1]
        state = 1
    elseif delta == [-1, 0]
        state = 2
    elseif delta == [1, 0]
        state = 3
    end
    return state
end

function randomstate(n::Int64)
    A = set_lattice(n)
    g = dimersample(div(n, 2))

    M = zeros(Int8, n, n)

    for i=1:div(n, 2)
        for j=1:div(n, 2)
            k = 2i-1
            l = 2j
            
            new = [t for t in Graphs.out_neighbors((k, l), g)[1]]
            delta1 = [k, l] - new

            state = delta2state(delta1)
            M[k, l] = state
            M[new[1], new[2]] = invert(state)
            
            u = 2i
            v = 2j-1
            
            new = [t for t in Graphs.out_neighbors((u, v), g)[1]]
            delta2 = [u, v] - new
            
            state = delta2state(delta2)
            M[u, v] = state
            M[new[1], new[2]] = invert(state)
        end
    end
    set_state!(A, M, n)
    return A, M
end