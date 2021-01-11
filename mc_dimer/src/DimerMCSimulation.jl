"""
Author: Doruk Efe Gökmen
Date: 24/05/2020

TODO: Implement memory efficient configuration saver.
TODO: Generate random initial dimer configurations using Dimers.jl package.
"""

using Distributed
using Printf
using Glob
using Dates
using ArgParse
using Parsers
using MCMCDiagnostics
using PyCall
np = pyimport("numpy") # for saving samples

# -1. Take arguments from command line and parse them.
function parse_commandline()
	s = ArgParseSettings()

	s.prog = "DimerMCSimulation"
	s.description = "Run the directed loop MC simulation at given temperature and system size."

	@add_arg_table! s begin
		"--basedir", "-b"
			help = "Base directory. Enter string."
			default = "/cluster/home/user"
	   	"--outdir", "-d"
	   		help = "Output directory. Enter string."
	   		default = "/cluster/scratch/user/data"
   		"--sample-prefix", "-f"
   			help = "Prefix of output files. Enter string."
   			default = "configs_intdimer2d_square"
   		"--num-realisations", "-R"
			help = "Number of random initial conditions to be generated. Enter integer."
			arg_type = Int
			default = 2000
		"--num-sweeps", "-N"
			help = "Number of samples to be generated. Enter integer."
			arg_type = Int
			default = 200
		"--num-subsweeps", "-S"
			help = "Number of samples thrown away between samples with no autocorrelation."*
					"Enter integer."
			arg_type = Int
			default = 2000
   		"--eqtime", "-q"
   			help = "Equilibration times. Enter list, integer or string."
   			arg_type = Int
   			default = 5000
		"--mc-method", "-M"
			help = "The method for deriving MC transition probabilities."*
					"Enter 'minimal_bounce' or 'heat_bath'"
			default = "minimal_bounce"
		"--lp-method", "-l"
			help = "Linear programming solver for finding MC transition probabilities"*
					"with minimal bounce likelihoods. Enter one of"*
					"'GLPK', 'Clp'."
			default = "GLPK"
   		"--dimer-interaction", "-V"
   			help = "Aligning or antialigning interaction for dimers. Enter '-1' or '1'."
   			default = "-1"
		"--site-correlation", "-c"
			help = "Switches correlation between spurious sites at vertices and faces."
			action = :store_true
		"--temperatures", "-T"
			help = "List of temperatures. Enter list."
			arg_type = Float64
			default = 0.65
		"--system-size", "-n"
			help = "Linear system size. Enter integer."
			arg_type = Int64
			default = 20
		"--parallel", "-p"
			help = "Switch on parallelisation."
			action = :store_true
		"--threads", "-t"
			help = "Number of threads for parallel computation. Enter integer."
			arg_type = Int64
			default = nprocs()
		"--verbose", "-v"
			help = "Switch on verbose output."
			action = :store_true
		"--debug", "-D"
			help = "Switch on debugging output."
			action = :store_true
	end
	return parse_args(s)
end

parsed_args = parse_commandline()

N_threads = parsed_args["threads"]

if parsed_args["parallel"]
	addprocs(N_threads - nprocs())
end

parsed_args = parse_commandline()

base_directory = parsed_args["basedir"]
@eval @everywhere base_directory=$base_directory
out_directory = parsed_args["outdir"]
samples_prefix = parsed_args["sample-prefix"]

MC_method = parsed_args["mc-method"]
LP_method = parsed_args["lp-method"]

N_reals = parsed_args["num-realisations"]
N_sweeps = parsed_args["num-sweeps"]
N_subsweeps = parsed_args["num-subsweeps"]

N_therm = parsed_args["eqtime"]
"""
eqtime = tryparse(Int, parsed_args["eqtime"])
if eqtime === nothing
	autocorr_dir = parsed_args["eqtime"]
else
	N_therm = eqtime
	autocorr_dir = nothing
end
"""

v = parse(Int, parsed_args["dimer-interaction"])

site_corr = parsed_args["site-correlation"]

T = parsed_args["temperatures"]

n = parsed_args["system-size"]

verbose = parsed_args["verbose"]
debug = parsed_args["debug"]

if verbose
	println(parsed_args)
end

# 0. Import dependencies and initialise configurations.

t0 = Dates.now()

if verbose
	println("Importing and precompiling dependencies...")
end

@everywhere using ProgressBars
@everywhere using SharedArrays
@everywhere using StatsBase
@everywhere using Statistics
@everywhere include(joinpath(base_directory, "RSMI-NE/mc_dimer/src/DimerDirectedLoop.jl"))

if verbose
	println("Precompilation complete.\n")
	println("Initialising the dimer configuration...")
end

"""
M = zeros(Int8, n, n)
for i in 1:n
    for j in 1:n
        if j % 2 == 1
            M[i,j] = 1
        end
    end
end

A = DimerDirectedLoop.set_lattice(n)
DimerDirectedLoop.set_state!(A, M, n)
"""

Xs = SharedArray(zeros(N_sweeps*N_reals, 2n, 2n))

DSB_values = SharedArray(zeros(N_sweeps*N_reals))
PSB_values = SharedArray(zeros(N_sweeps*N_reals))
plaquette_op_values = SharedArray(zeros(N_sweeps*N_reals))
Energy_values = SharedArray(zeros(N_sweeps*N_reals))

all_worm_lengths = SharedArray(zeros(N_sweeps*N_reals))
all_N_worms = SharedArray(zeros(N_sweeps*N_reals))


# 1. Perform sampling.

if verbose
	println("Initialisation complete.\n")
	println("Start sampling...")
end


ground_state_counts = SharedArray(zeros(N_sweeps*N_reals, 4))


@sync begin
	@distributed for r in 0:(N_reals-1)
		if verbose
			println("Generating random dimer configuration using Wilson algorithm...")
		end
		A, M = DimerDirectedLoop.randomstate(n)
		for _ in 1:10
	        DimerDirectedLoop.DLMC_step!(A, 10000.0, -1, n, "heat_bath")
	    end

		if verbose
			println("Random dimer configuration generated.\n")
			println("Start thermalisation...")
		end
		for _ in ProgressBar(1:N_therm) # thermalisation
			DimerDirectedLoop.DLMC_step!(A, T, v, n, MC_method, LP_method)
		end
		if verbose
			println("Thermalisation complete.\n")
			println("Start sweeping...")
		end

		global worm_lengths = nothing

		for t in ProgressBar(1:N_sweeps)
			for _ in 1:N_subsweeps
				_, worm_lengths = DimerDirectedLoop.DLMC_step!(A, T, v, n, MC_method, LP_method)
			end

			for nn in 1:4
				ground_state_counts[t+r*N_sweeps, nn] += A[11, nn].J #gather gs statistics
			end

			Xs[t+r*N_sweeps, :, :] = DimerDirectedLoop.sites_bonds(A, n, site_corr)

			all_worm_lengths[t+r*N_sweeps] = mean(worm_lengths)
			all_N_worms[t+r*N_sweeps] = mean(length(worm_lengths))
			#DSB_values[t+r*N_sweeps] = DimerDirectedLoop.DSB(A, n)
			#plaquette_op_values[t+r*N_sweeps] = DimerDirectedLoop.plaquette_op(A, n)
			Energy_values[t+r*N_sweeps], PSB_values[t+r*N_sweeps] = DimerDirectedLoop.energy_PSB(A, n, v)
		end
	end
end


if verbose
	println("Sampling complete.\n")
end

t1 = Dates.now()

open(joinpath(out_directory, "sample_details_L$(2n)_T$(@sprintf("%.3f", T)).txt"), "a") do io
    println(io, "Sampling complete. T = ", T)
    println(io,"effective_sample_size = ", effective_sample_size(Energy_values))
    println(io, "ground_state_counts = ", sum(ground_state_counts, dims=1))
    println(io, "mean number of worms per subsweep = ", mean(all_N_worms))
    println(io, "mean size of worms = ", mean(all_worm_lengths))
    println(io, "total_runtime = ", t1 - t0, "\n")
end

np.save(joinpath(out_directory, samples_prefix*"_L$(2n)_T$(@sprintf("%.3f", T))"), Xs)

#np.save(joinpath(out_directory, "DSB_intdimer2d_square"*"_L$(2n)_T$(@sprintf("%.3f", T)).npy"), DSB_values)
#np.save(joinpath(out_directory, "PSB_intdimer2d_square"*"_L$(2n)_T$(@sprintf("%.3f", T)).npy"), PSB_values)
np.save(joinpath(out_directory, "E_intdimer2d_square"*"_L$(2n)_T$(@sprintf("%.3f", T)).npy"), Energy_values)
np.save(joinpath(out_directory, "P_intdimer2d_square"*"_L$(2n)_T$(@sprintf("%.3f", T)).npy"), plaquette_op_values)


