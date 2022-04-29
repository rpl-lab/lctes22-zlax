help:
	@echo "Usage:"
	@echo "  make init         # install Zelus, ProbZelus, and Zlax/ProbZlax"
	@echo "  make test         # run an example (more examples are available in the examples directory)"
	@echo "  make test_bench   # run a scaled down version of the benchmarks"

init: install_zelus install_probzelus install_zlax install_bench

install_zelus:
	opam pin -y -k path zelus

install_probzelus:
	opam pin -y -k path probzelus/zelus-libs
	opam pin -y -k path probzelus/probzelus

install_zlax:
	opam pin -y -k path zlax
	pip install ./zlax/zlax
	pip install ./zlax/probzlax

install_bench:
	opam install -y csv mtime

test:
	$(MAKE) -C examples coin

test_bench:
	$(MAKE) -C zlax-benchmarks zlax_build
	$(MAKE) -C zlax-benchmarks NUMRUNS=3 MIN=100 MAX=500 zlax_bench
	$(MAKE) -C zlax-benchmarks zlax_analyze
	$(MAKE) -C zlax-benchmarks zlax_plot

.PHONY: help init install_zelus install_probzelus install_zlax install_bench \
	test test_bench
