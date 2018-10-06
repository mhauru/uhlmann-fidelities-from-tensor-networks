tar_name=uhlmann_code.tar.gz


########################################################################


all: paper

paper: data plots

practice: practice_data practice_plots


########################################################################

data: data_phases data_convergence data_quench_uhlmann data_quench_magnetization

data_phases:
	python3 uhlmann_compare.py -c confs/conf_umps.yaml -y\
		"h1: 1.0"\
		"h2: 1.01"\
		"chi1: 51"\
		"chi2: 51"\
		"L: 800"\
		"do_separate: True"\
		"do_exact: True"

data_convergence:
	python3 uhlmann_compare.py -c confs/conf_umps.yaml -y\
		"h1: 1.0"\
		"h2: 1.0"\
		"chi1: 21"\
		"chi2: 11"\
		"L: 300"\
		"do_separate: True"\
		"do_exact: True"
	python3 uhlmann_compare.py -c confs/conf_umps.yaml -y\
		"h1: 1.05"\
		"h2: 1.05"\
		"chi1: 21"\
		"chi2: 11"\
		"L: 300"\
		"do_separate: True"\
		"do_exact: True"

data_quench_uhlmann:
	python3 quench_uhlmann.py -c confs/conf_mcmps_timeevolution.yaml

data_quench_magnetization:
	python3 quench_magnetization.py -c confs/conf_mcmps_timeevolution.yaml

plots: plots_phases plots_convergence plots_quench_uhlmann plots_quench_magnetization


########################################################################


plots_convergence:
	python3 plot_uhlmann_convergence.py noshow

plots_quench_magnetization:
	python3 plot_quench_magnetization.py noshow

plots_quench_uhlmann:
	python3 plot_quench_uhlmann.py noshow

plots_phases:
	python3 plot_uhlmann_phases.py noshow


########################################################################
# These "practice" options run the same code, but with lower bond dimensions
# and a quick cut-off for how many iterations to allow for in optimizing the
# MPS ground state. It is used for testing the whole infrastructure.

practice_data: practice_data_phases practice_data_convergence practice_data_quench_uhlmann practice_data_quench_magnetization

practice_data_phases:
	python3 uhlmann_compare.py -c confs/conf_umps.yaml -y\
		"max_counter: 200"\
		"max_subcounter: 10"\
		"h1: 1.0"\
		"h2: 1.01"\
		"chi1: 21"\
		"chi2: 21"\
		"L: 800"\
		"do_separate: True"\
		"do_exact: True"

practice_data_convergence:
	python3 uhlmann_compare.py -c confs/conf_umps.yaml -y\
		"max_counter: 200"\
		"max_subcounter: 10"\
		"h1: 1.0"\
		"h2: 1.0"\
		"chi1: 21"\
		"chi2: 11"\
		"L: 300"\
		"do_separate: True"\
		"do_exact: True"
	python3 uhlmann_compare.py -c confs/conf_umps.yaml -y\
		"max_counter: 200"\
		"max_subcounter: 10"\
		"h1: 1.05"\
		"h2: 1.05"\
		"chi1: 21"\
		"chi2: 11"\
		"L: 300"\
		"do_separate: True"\
		"do_exact: True"

practice_data_quench_uhlmann:
	python3 quench_uhlmann.py -c confs/conf_mcmps_timeevolution.yaml -y\
		"mps_chis: !!python/object/apply:builtins.range [1, 21, 1]"\
		"max_counter: 200"\
		"max_subcounter: 10"

practice_data_quench_magnetization:
	python3 quench_magnetization.py -c confs/conf_mcmps_timeevolution.yaml -y\
		"mps_chis: !!python/object/apply:builtins.range [1, 21, 1]"\
		"max_counter: 200"\
		"max_subcounter: 10"


########################################################################


practice_plots: practice_plots_phases practice_plots_convergence practice_plots_quench_uhlmann practice_plots_quench_magnetization

practice_plots_convergence:
	python3 plot_uhlmann_convergence.py noshow practice

practice_plots_quench_magnetization:
	python3 plot_quench_magnetization.py noshow practice

practice_plots_quench_uhlmann:
	python3 plot_quench_uhlmann.py noshow practice

practice_plots_phases:
	python3 plot_uhlmann_phases.py noshow practice

########################################################################

clean_all: clean_logs clean_plots clean_data
	
clean_data:
	rm -rf data uhlmann_compare_data quench_magnetization_data quench_uhlmann_data

clean_logs:
	rm -rf logs

clean_plots:
	rm -f *.pdf

clean_source_tar:
	rm -f ${tar_name}

########################################################################

source_tar: clean_source_tar
	tar czf ${tar_name}\
		MPS ncon tensors tntools\
		quench_magnetization.py uhlmann_compare.py quench_uhlmann.py\
		plot_uhlmann_convergence.py plot_quench_magnetization.py\
		plot_quench_uhlmann.py plot_uhlmann_phases.py\
		confs makefile\
		README LICENSE

