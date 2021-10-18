# Artefact JFLA 2022

Cet artefact correspond l'article "Inférence Parallèle pour un Langage Réactif Probabiliste" soumis aux JFLA 2022.

_G. Baudart, L. Mandel, M. Pouzet, R. Tekin_

## Installation

Les prérequis pour l'installation de l'artefact sont :
- [opam](http://opam.ocaml.org/) avec la version 4.13.1 d'OCaml
- [pip](https://pypi.org/project/pip/) avec la version 3.9 (ou plus récente) de Python


Les dépendences suivantes sont contenus dans les sous-modules de cet artifact :
- [Zelus](https://github.com/inria/zelus/tree/muf) (branche muf)
- [ProbZelus](https://github.com/IBM/probzelus)
- [Zlax/ProbZlax](https://github.com/rpl-lab/zlax)

Pour cloner le dépôt et installer toutes les dépendances :
```
$ git clone --recurse-submodules https://github.com/rpl-lab/jfla22-zlax.git
$ make init
```

Si tout se passe bien, cette commande installe le compilateur ProbZelus et le package python zlax qui contient, en particulier, le simulateur `zluciole`.
Les commandes suivantes permettent de tester l'installation.

```
$ probzeluc -version
$ zluciole --help
```

## Premiers pas

L'outil `zluciole` compile un programme ProbZelus en Python/JAX et simule son exécution.
Par exemple le fichier `examples/counter.zls` contient un unique nœud qui implémente un simple compteur.

```
let node main () = o where
    rec o = 0 -> (pre o + 1)
```

L'outil `zluciole` prend en argument le nombre d'instant à executer (option `-n`), le nœud à simuler (option `-s`) et le nom du fichier.
Le nœud simuler doit être de type `unit -> 'a` et ne pas faire appel à des fonctions non pures (qui ne peuvent pas être compilées en JAX).
L'outils `zluciole` imprime la sortie, une ligne par instant, sur la sortie standard.

```
$ zluciole -n 10 -s main examples/counter.zls
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
0
1
2
3
4
5
6
7
8
9
```

Le `WARNING` confirme l'utilisation de JAX.
Si CUDA n'est pas installé, JAX execute le programme sur CPU.

Les instructions pour installer la version de JAX compatible avec GPU sont disponibles [ici](https://github.com/google/jax#installation).

## Inférence réactive parallèle

Le dossier `examples` contient les deux exemples présentés dans le papier.
Pour simuler un programme probabiliste il faut préciser à `zluciole` d'utiliser les modules d'inférence (option `-prob`).

Le `Makefile` du dossier `examples` permets d'exécuter rapidement ces exemples.

```
$ cd examples
$ make
Choose one of coin, hmm, or hmm-plot.
```

## Exemple 1: `coin`

Le programme `coin.zls` lève une alarme si lorsqu'on détecte qu'une pièce est trop biaisée à partir d'observations statistiques.
Dans cet exemple, on suppose que les observations sont toujours `true`/pile (l'entrée du nœud `cheater_detector` est la constante `true` dans le nœud `main`).

```
node watch x = alarm where
  rec automaton
  | Wait -> do alarm = false until x then Ring
  | Ring -> do alarm = true done

proba coin obs = theta where
    rec init theta = sample (uniform_float (0.0, 1.0))
    and () = observe (bernoulli theta, obs)

node cheater_detector x = cheater, (m, s) where
  rec theta_dist = infer 1000 coin x
  and m, s = stats_float theta_dist
  and cheater = watch ((m < 0.2 || 0.8 < m) && (s < 0.01))

node main () = ("cheater", cheater), ("mean", m), ("std", s) where
    rec cheater, (m, s) = cheater_detector true
```

On peut lancer ce programme avec la commande suivante (`make coin`) :

```
$ zluciole -prob -n 20 -s main coin.zls
(('cheater', False), ('mean', 0.6717270016670227), ('std', 0.11130374670028687))
(('cheater', False), ('mean', 0.7557703256607056), ('std', 0.06814475357532501))
(('cheater', False), ('mean', 0.810149610042572), ('std', 0.04151570796966553))
(('cheater', False), ('mean', 0.8406215906143188), ('std', 0.02981172874569893))
(('cheater', False), ('mean', 0.8691133260726929), ('std', 0.019973836839199066))
(('cheater', False), ('mean', 0.8866546154022217), ('std', 0.014147769659757614))
(('cheater', False), ('mean', 0.8979821801185608), ('std', 0.010037768632173538))
(('cheater', False), ('mean', 0.9038760662078857), ('std', 0.008628054521977901))
(('cheater', True), ('mean', 0.9117444157600403), ('std', 0.007092374376952648))
(('cheater', True), ('mean', 0.9166853427886963), ('std', 0.00595330772921443))
```


Au bout de 9 instants, la condition `(m < 0.2 || 0.8 < m) && (s < 0.01)` sur la moyenne `m` et la variance `s` de la distribution sur le biais inféré devient vraie.
L'alarme `cheater` est alors levée.


## Exemple 2 : HMM

Le programme `hmm.zls` implémente un simple traqueur de position (à une dimension).
À chaque instant on suppose que la position courante suit une distribution normale autour de la position précédente, et que l'observation courante suit une distribution normale autour de la position courante.


```
proba hmm  obs = x where
  rec x = sample (gaussian (0.0 -> pre x, speed))
  and () = observe (gaussian (x, noise), obs)

node main () = t, obs, m, s where
    rec t = 0. fby (t +. 0.1)
    and obs = 10.0 *. sin(t) +. (draw (gaussian (0.0, 1.0)))
    and x_dist = infer 1000 hmm obs
    and m, s = stats_float x_dist
```

Comme pour le modèle précédent, on peut lancer ce programme avec la commande suivante pour obtenir à chaque instant, la date courante (incrementée de 0.1 à chaque instant), l'observation courante, la moyenne de la position estimée et sa déviation standard (`make hmm`):

```
$ zluciole -prob -n 10 -s main hmm.zls 
(0.0, -0.6177732944488525, -0.21219289302825928, 2.1207327842712402)
(0.10000000149011612, 1.695770025253296, 0.6171027421951294, 4.265596866607666)
(0.20000000298023224, 1.8255577087402344, 1.1918796300888062, 4.415806293487549)
(0.30000001192092896, 0.6059656143188477, 0.9084848165512085, 4.367940902709961)
(0.4000000059604645, 4.3959431648254395, 2.6987271308898926, 7.60023832321167)
(0.5, 3.355968475341797, 3.0306355953216553, 4.389452934265137)
(0.6000000238418579, 4.826956748962402, 3.942495822906494, 4.642572402954102)
(0.7000000476837158, 7.527453899383545, 5.652518272399902, 6.94222354888916)
(0.8000000715255737, 6.175825119018555, 5.990218639373779, 4.096563816070557)
(0.9000000953674316, 6.605088710784912, 6.298244953155518, 3.929673194885254)
```

On peut ensuite rediriger cette sortie vers gnuplot pour obtenir une représentation graphique (`make hmm-plot`).

![fig-hmm](./examples/fig-hmm.svg | width=150)