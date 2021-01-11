"""
Author: Doruk Efe Gökmen
Date: 07/04/2020

Perform a Monte Carlo update on (interacting) dimer configurations
either via the heat bath solution of the condition of detailed balance
or via the minimal bounce solution obtained via linear programming.
"""

module DimerDirectedLoop

export DLMC_step!, sites_bonds

using StatsBase

include("Lattice.jl")
include("RandomDimers.jl")
include("DimerMove.jl")
include("MinimalBounceTransitions.jl")
include("Measurements.jl")
include("DrawDimerConfigs.jl")


function DLMC_step!(A::Array{Neighbour{Int8},2}, 
    T::Float64, v::Int, n::Int, 
    MC_method::String="minimal_bounce",
    LP_pack::String="GLPK", precompute_matrices::Bool=true)
    """
    Constructs a single worm loop at temperature T 
    on a system with linear size n interacting with 
    aligning or anti-aligning Hamiltonian according to v.
    """
    N = n^dim # system size

	if precompute_matrices 
	# precompute the MC scattering matrices for all possible nn orientation statistics:
	    m = [[2,0,1,0], [2,0,0,1], [2,0,0,0],
		     [0,2,1,0], [0,2,0,1], [0,2,0,0],
		     [1,0,2,0], [0,1,2,0], [0,0,2,0],
		     [1,0,0,2], [0,1,0,2], [0,0,0,2]]
		t = reverse.(Iterators.product(fill(0:1, 4)...))[:]
		all_counts = vcat([collect(i) for i in t],m)

		all_MCS_matrices = Dict() #Array{Array{Float64,1},1}
		for counts in all_counts
			weights = Array{Float64,1}(UndefInitializer(), D)
			for nn_j in 0:D-1
			    weights[nn_j+1] = exp(-v*counts[nn_j+1]/T)
			end
			all_MCS_matrices[counts] = MCS_matrix(weights)
		end
	end


    visited = []
    worm_lengths = []

    while length(visited) < N

        i_0 = rand(1:N) # worm entry index
        global i = i_0 # initial entrance index
        global k = nothing # initialise exit index
        
        global nn_new = 0

        global worm = [] # initialise list containing vertices that worm passes

        while k != i_0
            if k == nothing
                nn = get_dimer(A, i)[1] # initial connected neighbour
            else
                nn_defect = get_dimer(A, i) # all dimers connected to i
                
                # follow existing dimer instead of one just added
                filter!(e -> e ≠ invert(nn_new), nn_defect)
                nn = nn_defect[1]
            end
            
            j = A[i, nn+1].site # initial pivot index
            
            nn_entry = invert(nn)
            
            
            # disconnect i and its nn'th neighbour
            set_dimer!(A, j, nn_entry, Int8(0))
            
            # perform Monte Carlo update:

            if MC_method == "minimal_bounce"
	            if precompute_matrices
	            	parallel_counts = compute_parallel_counts(A, j) # count nn orientations
	                S = all_MCS_matrices[parallel_counts] # get precomputed MCS matrix
	            else
		            weights = compute_weights(A, j, T, v) # dimer weights
	                S = MCS_matrix(weights, LP_pack) # compute scattering matrix
                end
                transition_probs = [S[nn_entry+1, nn_exit+1] for nn_exit in 0:D-1]
                
            elseif MC_method == "heat_bath"
                weights = compute_weights(A, j, T, v) # dimer weights
	            transition_probs = [w/sum(weights) for w in weights]
	        end
        
            nn_new = sample([l for l in 0:D-1], Weights(transition_probs))
            
            # connect j and its nn_new'th neighbour
            set_dimer!(A, j, nn_new, Int8(1))
            
            k = A[j,nn_new+1].site # exit index

            append!(worm, [(i,j,k)]) # save worm indices for future use
            append!(visited, j)

            i = k # new entry index
        end
        visited = unique(visited)
        append!(worm_lengths, length(worm))
    end
    return worm, worm_lengths
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