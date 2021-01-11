"""
Author: Doruk Efe Gökmen
Date: 06/04/2020

Monte Carlo measurement of thermodynamic quantities from dimer samples.
"""

export hv_dimer_count, hv_plaquette_count
export DSB, energy_PSB, plaquette_op
export G_dd_l, G_dd_t
export G_mm

include("Lattice.jl")


function hv_dimer_count(A::Array{Neighbour{Int8},2}, n::Int)
    """
    Return the count of horizontal and vertical dimers in configuration A.
    """
    N = n^dim # system size
    vertices = [i for i in 1:N]

    count_h = 0 
    count_v = 0
    while length(vertices) > 0
        found_dimer = false

        i = vertices[1]
        filter!(x->x≠i, vertices) # remove current vertex from list

        for h in 0:1 # count horizontal dimers
            dimer_h = A[i, h+1].J
            count_h += dimer_h
            if dimer_h == 1
                filter!(x->x≠A[i, h+1].site, vertices)
                found_dimer = true
            end
        end
        if found_dimer == false # if site i is not connected horizontally
            for v in 2:3 # count vertical dimers
                dimer_v = A[i, v+1].J
                count_v += dimer_v
                if dimer_v == 1
                    filter!(x->x≠A[i, v+1].site, vertices)
                end
            end
        end
    end
    return count_h, count_v
end


function hv_plaquette_count(A::Array{Neighbour{Int8},2}, n::Int)
    """
    Return the count of plaquettes containing two parallel horizontal dimers
    and two parallel vertical dimers in configuration A.
    """
    N = n^dim # system size

    count_h = 0 
    count_v = 0

    for i in 1:N
        if A[i, 1].J == 1
            count_h += A[A[i, 3].site, 1].J
        elseif A[i, 3].J == 1
            count_v += A[A[i, 1].site, 3].J
        end
    end
    return count_h, count_v
end


function DSB(A::Array{Neighbour{Int8},2}, n::Int)
    """
    Dimer rotation symmetry breaking order parameter.
    """
    N = n^dim # system size
    count_h, count_v = hv_dimer_count(A, n)
    return abs(count_h - count_v)/N
end


function energy_PSB(A::Array{Neighbour{Int8},2}, n::Int, v::Int)
    """
    Dimer pair rotation symmetry breaking order parameter and
    the interaction energy of the system.
    """
    N = n^dim # system size
    count_h, count_v = hv_plaquette_count(A, n)

    Energy = v*(count_h + count_v)
    PSB = abs(count_h - count_v)/N
    return Energy, PSB
end


function plaquette_op(A::Array{Neighbour{Int8},2}, n::Int)
    """
    Order parameter to discriminate the plaquette phase.
    """
    N = n^dim # system size

    P = 0
    for i in 0:N-1
        k = div(i,n)
        l = i % n

        if A[i+1, 1].J == 1
            P += (-1)^(k+l) * A[A[i+1, 3].site, 1].J
        elseif A[i+1, 3].J == 1
            P += (-1)^(k+l) * A[A[i+1, 1].site, 3].J
        end
    end
    return abs(P)/N
end


function G_dd_l(A::Array{Neighbour{Int8},2}, n::Int)
    """
    Longitudinal (horizontal) dimer-dimer correlator.
    TODO: implement
    """

end


function G_dd_t(A::Array{Neighbour{Int8},2}, n::Int)
    """
    Transverse (horizontal) dimer-dimer correlator.
    TODO: implement
    """

end


function G_mm(A::Array{Neighbour{Int8},2}, n::Int)
    """
    Monomer-monomer correlator.
    Improved estimator computed during worm construction.
    TODO: implement
    """

end