init: install_zelus install_probzelus install_zlax
	

install_zelus:
	opam pin -y zelus https://github.com/INRIA/zelus.git#17415dd9a816c1b2cec75d5024c6dbe38a76ef6d


install_probzelus:
	opam pin -y -k path probzelus/zelus-libs
	opam pin -y -k path probzelus/probzelus

install_zlax:
	opam pin -y https://github.com/rpl-lab/zlax.git#7b0aec07e37560b1686c644bdb855191649c3c20
	pip install git+git://github.com/rpl-lab/zlax.git@7b0aec07e37560b1686c644bdb855191649c3c20
