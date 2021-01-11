"""
Author: Doruk Efe Gökmen
Date: 02/04/2020

Obtain Monte Carlo scattering matrix that minimises likelihood of bounce events
by solving a linear programming problem arising from 
a constraint detailed balance condition.
"""

export MCS_matrix

using JuMP
#using CPLEX
#using Gurobi
using GLPK
using Clp

include("Lattice.jl")

LP_packs = Dict("GLPK"=>GLPK, "Clp"=>Clp) #,"CPLEX"=>CPLEX, "Gurobi"=>Gurobi)

function MCS_matrix(w::Array{Float64,1}, LP_pack::String="GLPK")
    #TT = stdout # save original STDOUT stream

    LP_pack_module = LP_packs[LP_pack]
    
    A_ub = [1 1 1 0 0 0;
            1 0 0 1 1 0;
            0 1 0 1 0 1;
            0 0 1 0 1 1]   #A_ub_matrix(w)

    c = [1 1 1 1 1 1] #c_vector(w)
    
    num_vars = Int(D * (D-1)/ 2)
        
    m = Model()
    set_optimizer(m, LP_pack_module.Optimizer)

    @variable(m, 0<=a[1:num_vars])
    @objective(m, Min, -sum(c[i]*a[i] for i in 1:num_vars))
    @constraint(m, constraint[j=1:D], sum(A_ub[j,i]*a[i] for i=1:num_vars) <= w[j])
    
    #redirect_stdout()
    optimize!(m)
    #redirect_stdout(TT) 
    
    RL, RB, RU, LB, LU, BU = [JuMP.value(a[i]) for i in 1:num_vars]
        
    RR = w[1] - (RL + RB + RU)
    LL = w[2] - (RL + LB + LU)
    BB = w[3] - (RB + LB + BU)
    UU = w[4] - (RU + LU + BU)
    
    S = [RR/w[1] RL/w[1] RB/w[1] RU/w[1];
         RL/w[2] LL/w[2] LB/w[2] LU/w[2];
         RB/w[3] LB/w[3] BB/w[3] BU/w[3];
         RU/w[4] LU/w[4] BU/w[4] UU/w[4]]
    return S
end

