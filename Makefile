init: install_zelus install_probzelus install_zlax
	

install_zelus:
	opam pin -y -k path zelus

install_probzelus:
	opam pin -y -k path probzelus/zelus-libs
	opam pin -y -k path probzelus/probzelus

install_zlax:
	opam pin -y -k path zlax
	pip install ./zlax