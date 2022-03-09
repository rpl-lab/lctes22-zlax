help:
	@echo "Usage:"
	@echo "  make init     # install Zelus, ProbZelus, and Zlax/ProbZlax"
	@echo "  make -C coin  # run an example (more examples are available in the examples directory)"

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

.PHONY: help init install_zelus install_probzelus install_zlax install_bench
