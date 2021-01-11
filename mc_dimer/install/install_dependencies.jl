#Installs the dependencies for the DimerDirectedLoop.jl module.

using Pkg

Pkg.add("JuMP")
Pkg.add("GLPK")
Pkg.add("Clp")
Pkg.add("StatsBase")
Pkg.add("Statistics")
Pkg.add("Distributed")
Pkg.add("SharedArrays")
Pkg.add("PyCall")
Pkg.add("Glob")
Pkg.add("ArgParse")
Pkg.add("Parsers")
Pkg.add("ProgressBars")