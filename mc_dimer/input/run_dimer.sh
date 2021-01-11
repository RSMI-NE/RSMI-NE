julia ../src/DimerMCSimulation.jl --basedir "/RSMI-NE/" --outdir "/RSMI-NE/mc_dimer/data"\
							--num-sweeps 100 --num-realisations 100 --eqtime 60 --num-subsweeps 2\
							--temperatures 10.0 --system-size 30 --threads 5 -cpv
