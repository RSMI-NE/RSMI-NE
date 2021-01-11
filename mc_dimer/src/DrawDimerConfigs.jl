"""
Author: Doruk Efe Gökmen
Date: 06/04/2020

Draw the dimer configuations using matplotlib and networkx packages in Python.
"""

export draw_config
export draw_worm

using PyCall
nx = pyimport("networkx")
plt = pyimport("matplotlib.pyplot")

include("Lattice.jl")

function draw_config(A::Array{Neighbour{Int8},2}, n::Int, save::Bool)
    """
    Draws the dimer configuration as a graph.
    """
    N = n^dim # system size

    G = nx.Graph()

    for i in 0:N-1
        if div(i,n)==0 || div(i,n)==n-1
            continue
        elseif i%n==0 || i%n==n-1
            continue
        else
            for nn in 0:D-1
                j = A[i+1,nn+1].site - 1
                if A[i+1,nn+1].J==0
                    G.add_edge((i%n,n-div(i,n)), (j%n,n-div(j,n)), weight=0.2)
                else
                    G.add_edge((i%n,n-div(i,n)), (j%n,n-div(j,n)), weight=1)
                    
                end
            end
        end
    end

    elarge = [(u,v) for (u,v,d) in G.edges(data=true) if d["weight"]>0.5]
    esmall = [(u,v) for (u,v,d) in G.edges(data=true) if d["weight"]<=0.5]

    pos = Dict((n,n) for n in G.nodes())

    plt.figure(figsize=(div(n,3),div(n,3)))

    nx.draw_networkx_nodes(G, pos, node_size=7)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=5)
        
    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                            width=5, alpha=0.15, edge_color="b")

    plt.axis("off")
    plt.show()
    if save
        plt.savefig("dimer_config.pdf")
    end
end


function draw_worm(A::Array{Neighbour{Int8},2}, n::Int, worm, save::Bool)
    """
    Draws the dimer configuration and the input worm confuguration as a graph.
    """
    N = n^dim # system size

    G = nx.Graph()

    for i in 0:N-1
        if div(i,n)==0 || div(i,n)==n-1
            continue
        elseif i%n==0 || i%n==n-1
            continue
        else
            for nn in 0:D-1
                j = A[i+1,nn+1].site - 1
                if A[i+1,nn+1].J==0
                    G.add_edge((i%n,n-div(i,n)), (j%n,n-div(j,n)), weight=0.4)
                else
                    G.add_edge((i%n,n-div(i,n)), (j%n,n-div(j,n)), weight=1)
                    
                end
            end
        end
    end
    
    for tup in worm
        G.add_edge(((tup[1]-1)%n,n-div(tup[1]-1,n)), ((tup[2]-1)%n,n-div(tup[2]-1,n)), weight=0.2)
        G.add_edge(((tup[2]-1)%n,n-div(tup[2]-1,n)), ((tup[3]-1)%n,n-div(tup[3]-1,n)), weight=0.2)
    end

    elarge = [(u,v) for (u,v,d) in G.edges(data=true) if d["weight"]==1]
    esmall = [(u,v) for (u,v,d) in G.edges(data=true) if d["weight"]==0.4]
    eworm = [(u,v) for (u,v,d) in G.edges(data=true) if d["weight"]==0.2]

    pos = Dict((n,n) for n in G.nodes())

    plt.figure(figsize=(div(n,3),div(n,3)))

    nx.draw_networkx_nodes(G, pos, node_size=7)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=5)
    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                            width=5, alpha=0.15, edge_color="b")
    nx.draw_networkx_edges(G, pos, edgelist=eworm,
                            width=7, alpha=0.15, edge_color="r")

    plt.axis("off")
    plt.show()
    if save
        plt.savefig("dimer_config.pdf")
    end
end