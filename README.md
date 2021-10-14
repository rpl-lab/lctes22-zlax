# Artefact JFLA 2022

Cet artefact correspond l'article "Inférence Parallèle pour un Langage Réactif Probabiliste" soumis aux JFLA 2022.

## Installation

Les prérequis pour l'installation de l'artefact sont :
- [opam](http://opam.ocaml.org/) avec la version 4.13.1 d'OCaml
- [pip](https://pypi.org/project/pip/) avec la version 3.9 (ou plus récente) de Python


Les dépendences suivantes sont contenus dans les sous-modules de cet artifact :
- [Zelus](https://github.com/inria/zelus/tree/muf) (branche muf)
- [ProbZelus](https://github.com/IBM/probzelus)
- [Zlax/ProbZlax](https://github.com/rpl-lab/zlax)

Pour cloner le dépôt et installer toutes les dépendences :
```
git clone --recurse-submodules https://github.com/rpl-lab/jfla22-zlax.git
make init
```
