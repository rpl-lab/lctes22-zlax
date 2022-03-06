# LCTES 2022 Artefact

This artifact corresponds to the article **JAX Based Parallel Inference for Reactive Probabilistic Programming** submitted to LCTES 2022.

## Installation

The prerequisites for installing the artifact are:
- [opam](http://opam.ocaml.org/) with OCaml version 4.13.1
- [pip](https://pypi.org/project/pip/) with Python version 3.9 (or newer)

This artifact contains:
- `zelus`: a modified version of the [Zelus](https://zelus.di.ens.fr) compiler with a new JAX backend
- `probzelus`: the original [ProbZelus](https://github.com/IBM/probzelus) runtime for OCaml
- `zlax`: the new ProbZelus runtime for JAX
- `examples`: some examples of ProbZelus programs
- `zlax-benchmarks`: the benchmarks used for the evaluation.


The following commands install the ProbZelus compiler and the `zlax` Python package and allow to test the installation:

```
$ make init
$ probzeluc -version
$ zluciole --help
```

## First steps

The `zluciole` tool compiles a ProbZelus program in Python/JAX and drives its execution.
For example, the  file `examples/counter.zls` contains a node that implements a counter.

```
let node main () = o where
    rec o = 0 -> (pre o + 1)
```

The `zluciole` tool takes as argument the number of time steps to execute (the `-n` option), the node to execute (the `-s` option), and the name of the file.
The node to execute must have type `unit -> 'a` and can only call pure functions (functions with side effects cannot be compiled into JAX).
`zluciole` prints on the standard output the ouput of the program one line per instant.

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

The `WARNING` confimes the use of JAX.
If CUDA is not installed, JAX executes the program on CPU.

_Remark._ The instructions to install a version of JAX compatible with GPUs are available at https://github.com/google/jax#installation.

## Reactive and Parallel Inference

The `examples` directory contains the examples presented in Sections 2 and 3.
The `Makefile` in this directory allows to execute these examples.

```
$ cd examples
$ make
Help:
  make coin      # simulate the coin example
  make hmm       # simulate the hmm example
  make hmm-plot  # simulate the hmm example with graphical interface using gnuplot
```

### Example 1 : `coin`

The `coin.zls` program raises an alarm when it detects from statistical observations a coin which is too biased.
In this example, we assume that the observations are always `true`/head (the input of the `cheater_detector` node is the contant `true` in the main node).

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

To execute a probabilistic program, we must specify to `zluciole` the use of inference module (the `-prob` option).
The `coin` example can be executed with the following command (cf. `make coin`):

```
$ zluciole -prob -n 10 -s main coin.zls
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

After 9 time steps, the condition `(m < 0.2 || 0.8 < m) && (s < 0.01)` about the mean `m` and standard deviation `s` of the distribution on the biais `theta` becomes true.
The alarm `cheater` is then raised.


### Example 2 : `HMM`

The `hmm.zls` program implements a simple position tracker (with one dimension).
At each time step, we assume that the current position follows a normal distribution around the previous position, and the current obstervation follows a normal distribution around the current position.

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

As for the previous model, we can launch this program with the following command to obtain at each instant, the current date `t` (incremented by 0.1 at each instant), the current observation `obs`, the average `m` of the estimated position `x_dist` and its standard deviation `s` (cf .`make hmm`):


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

We can redirect this output to gnuplot to obtain a graphical representation (cf. `make hmm-plot`).

<img src="./examples/fig-hmm.svg" alt="fig-hmm" width=500>

# Benchmarks

The directory `zlax-benchmarks` contains the benchmarks used in Section 5 of the paper for the evaluation. To reproduce the benchmarks you can execute the following commands.

```
cd zlax-benchmarks     # go to the benchmarks directory
make zlax_build        # build all the examples
make zlax_bench        # run the experiments and produce the data in csv file in each sub-directories
make zlax_analyze      # analyse the csv files to produce some summary
make -C plot zlax_all  # generate the graphs  in `plot/*.png`
```

WARNING: the execution might take several days.

The scale of the experiments can be configure through makefile variables. For example the experiments can be launch for the range of 1000 to 5000 particles with only 3 runs per number of particles as follow:

```
make NUMRUNS=3 MIN=1000 MAX=5000 zlax_bench
```

The same command can be executed in any sub-directory to run only the selected example.
