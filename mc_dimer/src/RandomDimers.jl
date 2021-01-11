"""
[Based on arXiv:math/9903025v2 by Kenyon, Propp, Wilson (2000)]
Generates samples from the dimer model on a 2-d rectangular lattice.
Also supports loop erased random walks and Wilson's algorithm on an 
arbitrary graph.

Authors: Samuel S. Watson, Doruk Efe Gökmen
Date: 25/02/2020
"""

export Wilson, 
       LERW, 
       gridgraph, 
       rotate, 
       flatten, 
       midpoint, 
       dimersample, 
       dimerheight,
       dimerSitesBonds

import Graphs

function LERW(Γ::Graphs.AbstractGraph,
                   startingvertex::Int64,
                   roots::Array{Bool,1};
                   maxiter::Integer=10^7)
    """
    Loop erased random walk on the input graph Γ.
    """

    X = [startingvertex]
    cntr = 1
    while ~(roots[X[end]])
        neighbors = Γ.adjlist[X[end]]
        push!(X,Graphs.vertex_index(neighbors[rand(1:length(neighbors))],Γ))
        i = 1
        while X[i] != X[end]
            i += 1
        end
        X = X[1:i]
        cntr += 1
        if cntr >= maxiter
            error("Maximum iterations hit in LERW")
        end
    end
    vert = Graphs.vertices(Γ)
    return vert[X]
end


function Wilson(Γ::Graphs.AbstractGraph,roots::Array{Bool,1})

    if length(Graphs.connected_components(Γ)) > 1
        error("Graph not connected")
    end

    maxiter = length(Graphs.vertices(Γ))
    cntr = 1
    UST = Graphs.adjlist(typeof(Graphs.vertices(Γ)[1]))
    for v in Graphs.vertices(Γ)
        Graphs.add_vertex!(UST,v)
    end
    discovered = roots
    while ~all(discovered)
        i = 1
        while discovered[i]
            i += 1
        end
        lerw = LERW(Γ,i,discovered)
        for k=1:length(lerw)-1
            Graphs.add_edge!(UST,lerw[k+1],lerw[k])
            discovered[Graphs.vertex_index(lerw[k],Γ)] = true
        end
        cntr += 1
        if cntr >= maxiter
            error("Something's gone wrong with Wilson's algorithm")
            break
        end
    end
    return UST
end


rotate(segment::Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}) = 
(div(segment[1][1] + segment[2][1] - segment[1][2] + segment[2][2],2), 
div(segment[1][1] - segment[2][1] + segment[1][2] + segment[2][2],2)),
(div(segment[1][1] + segment[2][1] + segment[1][2] - segment[2][2],2), 
div(-segment[1][1] + segment[2][1] + segment[1][2] + segment[2][2],2))

flatten(e::Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}) = vcat(map(x->vcat(x...),vcat(e...))...)

midpoint(point1::Tuple{Int64,Int64},point2::Tuple{Int64,Int64}) = (div(point1[1] + point2[1],2),div(point1[2] + point2[2],2))

function gridgraph(n::Int64)

    Γ = Graphs.adjlist(Tuple{Int64,Int64},is_directed=false)

    for i=1:n
        for j=1:n
            Graphs.add_vertex!(Γ,(i,j))
        end
    end

    gridedges = Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}[]

    for i=1:n
        for j=1:n
            if i < n
                Graphs.add_edge!(Γ,(i,j),(i+1,j))
            end
            if j < n
                Graphs.add_edge!(Γ,(i,j),(i,j+1))
            end
        end
    end
    
    return Γ 
end

function dimersample(m::Int64,n::Int64)
    
    function add_edge_and_continue(vertex::Tuple{Int64,Int64},prev_vertex::Tuple{Int64,Int64})
        Graphs.add_edge!(dualtree_ordered,prev_vertex,vertex)
        if (vertex[1]+2,vertex[2]) != prev_vertex && (vertex[1]+2,vertex[2]) in Graphs.out_neighbors(vertex,dualtree)
            add_edge_and_continue((vertex[1]+2,vertex[2]),vertex)
        end
        if (vertex[1]-2,vertex[2]) != prev_vertex && (vertex[1]-2,vertex[2]) in Graphs.out_neighbors(vertex,dualtree)
            add_edge_and_continue((vertex[1]-2,vertex[2]),vertex)
        end
        if (vertex[1],vertex[2]+2) != prev_vertex && (vertex[1],vertex[2]+2) in Graphs.out_neighbors(vertex,dualtree)
            add_edge_and_continue((vertex[1],vertex[2]+2),vertex)
        end
        if (vertex[1],vertex[2]-2) != prev_vertex && (vertex[1],vertex[2]-2) in Graphs.out_neighbors(vertex,dualtree)
            add_edge_and_continue((vertex[1],vertex[2]-2),vertex)
        end
    end

    Γ = Graphs.adjlist(Tuple{Int64,Int64},is_directed=false)

    for i=1:m+1
        for j=1:n+1
            if (i,j) != (m+1,n+1)
                Graphs.add_vertex!(Γ,(2i-1,2j-1))
            end
        end
    end

    primaledges = Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}[]

    for i=1:m
        for j=1:n
            push!(primaledges,((2i-1,2j-1),(2i-1+2,2j-1)))
            push!(primaledges,((2i-1,2j-1),(2i-1,2j-1+2)))
        end
    end

    for e in primaledges
        Graphs.add_edge!(Γ,e...) 
    end

    dualtree = Graphs.adjlist(Tuple{Int64,Int64},is_directed=false)

    for i=0:m
        for j=0:n
            if (i,j) != (0,0)
                Graphs.add_vertex!(dualtree,(2i,2j))
            end
        end
    end

    roots = [false for i=1:length(Graphs.vertices(Γ))]
    dualroots = [false for i=1:length(Graphs.vertices(dualtree))]

    for i=1:m
        roots[Graphs.vertex_index((2i-1,2n+1),Γ)] = true
        dualroots[Graphs.vertex_index((2i,0),dualtree)] = true
    end

    for j=1:n
        roots[Graphs.vertex_index((2m+1,2j-1),Γ)] = true
        dualroots[Graphs.vertex_index((0,2j),dualtree)] = true
    end

    UST = Wilson(Γ,roots)

    for e in primaledges
        if ~(e[2] in Graphs.out_neighbors(e[1],UST) || e[1] in Graphs.out_neighbors(e[2],UST))
            newedge = rotate(e)
            Graphs.add_edge!(dualtree,newedge...)
        end
    end

    dualtree_ordered = Graphs.inclist(Tuple{Int64,Int64},is_directed=true)

    for v in Graphs.vertices(dualtree)
        Graphs.add_vertex!(dualtree_ordered,v) 
    end

    for k=2:2:2n
        if (2,k) in Graphs.out_neighbors((0,k),dualtree) || (0,k) in Graphs.out_neighbors((2,k),dualtree)
            add_edge_and_continue((2,k),(0,k))
        end
        if (k,0) in Graphs.out_neighbors((k,2),dualtree) || (k,2) in Graphs.out_neighbors((k,0),dualtree)
            add_edge_and_continue((k,2),(k,0))
        end    
    end

    dimergraph = Graphs.adjlist(typeof(Graphs.vertices(Γ)[1]))
    dimer_vertices = Tuple{Int64,Int64}[]
    dimer_edges = Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}[]

    for i=1:2m
        for j=1:2n
            push!(dimer_vertices,(i,j))
            if i < 2m
                push!(dimer_edges,((i,j),(i+1,j)))
            end
            if j < 2n
                push!(dimer_edges,((i,j),(i,j+1)))
            end
            Graphs.add_vertex!(dimergraph,(i,j))
        end
    end

    for v in Graphs.vertices(UST)
        for w in Graphs.out_neighbors(v,UST)
            Graphs.add_edge!(dimergraph,midpoint(v,w),w)
        end
    end

    for v in Graphs.vertices(dualtree_ordered)
        for w in Graphs.out_neighbors(v,dualtree_ordered)
            Graphs.add_edge!(dimergraph,midpoint(v,w),w)
        end
    end

    return dimergraph

end

dimersample(n::Integer) = dimersample(n,n)

function dimerSitesBonds(N_white::Int64, corr_spin::Bool)
    """
    TODO: Fix the bug in this function!
    """
    g = dimersample(N_white)

    A = zeros(Int8, 4N_white, 4N_white)

    for i=1:N_white
        for j=1:N_white
            k = 2i-1
            l = 2j
            delta1 = [t for t in Graphs.out_neighbors((k, l), g)[1]] - [k, l]
            
            m = 2i
            n = 2j-1
            delta2 = [t for t in Graphs.out_neighbors((m, n), g)[1]] - [m, n]
            
            A[([2k-1, 2l-1] + delta1)[1], ([2k-1, 2l-1] + delta1)[2]] = 1
            A[([2m-1, 2n-1] + delta2)[1], ([2m-1, 2n-1] + delta2)[2]] = 1
            
            for p=0:1
                for r=0:1
                    A[2(i+p*N_white)-1, 2(j+r*N_white)-1] = rand(0:1)
                    if corr_spin
                        A[2(i+p*N_white), 2(j+r*N_white)] = A[2(i+p*N_white)-1, 2(j+r*N_white)-1]
                    else
                        A[2(i+p*N_white), 2(j+r*N_white)] = rand(0:1)
                    end
                end
            end
        end
    end

    return A
end


function dimerheight(dimergraph::Graphs.AbstractGraph)
    
    all_edges = Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}[]
    
    m = div(maximum(map(x->x[1],Graphs.vertices(dimergraph))),2)
    n = div(maximum(map(x->x[2],Graphs.vertices(dimergraph))),2)

    for i=1:2m
        for j=1:2n
            if i < 2m
                push!(all_edges,((i,j),(i+1,j)))
            end
            if j < 2n
                push!(all_edges,((i,j),(i,j+1)))
            end
        end
    end

    down_edges = zeros(Bool,2m+1,2n+1)

    for i=1:2m-1
        for j=1:2n
            if (i,j) in Graphs.out_neighbors((i+1,j),dimergraph) || (i+1,j) in Graphs.out_neighbors((i,j),dimergraph)
                down_edges[i,j] = true
            end        
        end
    end

    h = zeros(Int64,2n+1,2n+1)

    for j=2:2:2n+1
        h[1,j] = 1
    end

    for i=2:2n+1
        h[i,1] = isodd(i) ? 0 : -1
        for j=2:2n+1
            if down_edges[i-1,j-1]
                h[i,j] = h[i-1,j] + (-1)^(i+j)
            else
                h[i,j] = h[i,j-1] + (-1)^(i+j+1)
            end
        end
    end
    
    return h
end


