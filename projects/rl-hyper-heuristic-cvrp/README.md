# RL Hyper-Heuristic for CVRP Search Operator Selection

A compact, reproducible research benchmark for **reinforcement-learning hyper-heuristics** on the Capacitated Vehicle Routing Problem (CVRP).

The learned policy does **not** construct routes, choose customers directly, or replace the routing heuristic. Its only decision is which low-level search operator to apply next:

```text
current feasible CVRP solution
        ↓
search-state features
        ↓
operator-selection policy
        ↓
classical low-level operator
        ↓
feasible candidate solution
        ↓
common acceptance rule
        ↓
next search state
```

This is an independent implementation. It is not a reproduction of a published codebase and does not claim paper-level performance or state-of-the-art results.

## Motivation

Classical local search and Adaptive Large Neighborhood Search (ALNS) depend strongly on which neighborhood or destroy/repair operator is applied at each stage of search. Standard adaptive mechanisms usually react to historical operator scores. Reinforcement learning provides a different hypothesis: use the **current search state** as context for operator selection.

Recent work has made this distinction explicit. Johnn et al. (2024) formulate ALNS operator choice as an MDP and learn an operator-selection policy, while Voigt (2025) shows through a large review that operator design and selection materially affect routing ALNS performance. The benchmark here studies the same control layer at a much smaller scale so that feasibility, randomization, exact small-instance checks, and paired comparisons remain auditable.

## Research Question

> Given a fixed pool of feasible CVRP search operators and the same initial solutions, acceptance rule, iteration budget, and random-seed protocol, can a reinforcement-learning selector use search-state information to improve final routing quality relative to fixed, random, round-robin, and classical adaptive operator-selection baselines?

The benchmark separates the following questions:

1. What is the OR problem? A synthetic Euclidean CVRP with vehicle-capacity constraints.
2. What does RL learn? Which **search operator** to invoke at the current iteration.
3. What does RL not learn? It does not emit a route, a customer permutation, or a vehicle assignment.
4. What remains classical? Initial construction, move generation, feasibility, objective evaluation, and simulated-annealing acceptance.
5. What is the primary decision metric? Best feasible final route distance, not Q-value accuracy or reward alone.
6. Where is exactness available? Small test instances are certified with an independent dynamic-programming CVRP oracle.

## Mathematical Problem

Let customer set be \(V=\{1,\ldots,n\}\), depot \(0\), vehicle capacity \(Q\), customer demand \(q_i\), and Euclidean edge cost \(d_{ij}\). A solution partitions customers into depot-to-depot routes \(R\). The benchmark minimizes

\[
\min_R \sum_{r\in R} \sum_{(i,j)\in r} d_{ij}
\]

subject to

\[
\sum_{i\in r} q_i \le Q \quad \forall r\in R,
\]

with every customer visited exactly once.

The RL selector never changes these constraints or the objective.

## Methodology

### Initial solution

Every method starts from the same deterministic nearest-feasible greedy construction for a given instance. The learned policy therefore cannot gain an advantage through a different constructor.

### Operator pool

Five low-level operators are implemented independently:

- `two_opt` — intra-route segment reversal;
- `swap` — customer exchange within or across routes, subject to capacity;
- `relocate` — move one customer to another position/route, subject to capacity;
- `ruin_recreate` — remove a customer subset and reinsert with best feasible insertion;
- `random_perturbation` — a short sequence of random relocate/swap moves for diversification.

Every returned candidate is independently audited. An operator that would violate customer uniqueness or capacity is not accepted as a valid search transition.

### Search state

The RL state contains normalized search-process information rather than raw customer identities:

- current objective relative to the initial objective;
- best-so-far objective ratio;
- recent normalized objective improvement;
- normalized stagnation length;
- recent acceptance ratio;
- remaining iteration budget;
- route-load coefficient of variation;
- route-length coefficient of variation;
- near-capacity-route fraction;
- route-count statistic;
- previous operator identity.

The first version uses a discretized state key and tabular Q-learning. This choice is deliberate: the research question is operator-selection control, not whether a large neural network can absorb instance geometry.

### Action

The action is one operator ID from the fixed operator pool. The policy cannot directly name a customer, route, edge, or permutation.

### Reward and final objective

The transition reward is

\[
r_t = \frac{f(x_t)-f(x_{t+1})}{f(x_0)},
\]

where \(x_{t+1}\) is the accepted next incumbent. Improving transitions are positive, accepted worsening transitions are negative, and rejected candidates receive zero.

This normalization changes the learning signal scale only. It does **not** change the CVRP objective. Final evaluation always uses best feasible routing distance. The benchmark does not report accumulated RL reward as a routing-quality metric.

### Common acceptance rule

All selectors use the same simulated-annealing acceptance rule and cooling schedule. Better candidates are always accepted; worsening candidates may be accepted according to the same temperature rule. Operator selection is therefore the main experimental difference.

Randomness is separated into deterministic streams for selector exploration, operator stochasticity, and acceptance draws. Methods receive the same instance and comparison seed. When two methods invoke the same operator at the same iteration, the operator-level random stream is aligned.

## Baselines

The evaluation panel contains:

- `fixed_best_validation` — the single fixed operator with the best mean normalized objective on validation instances, then frozen;
- `random` — uniform random operator selection;
- `round_robin` — deterministic cyclic operator selection;
- `adaptive_weights` — ALNS-style roulette selection with performance-reactive operator weights;
- `q_learning` — state-dependent RL operator selection.

The fixed operator is selected on validation only. Final test and OOD instances are never used to choose it.

## Evaluation Protocol

Train, validation, in-distribution test, and OOD size-shift instances use disjoint seed ranges. Instances are generated independently rather than as mutations of a shared base instance.

For every comparison seed and test instance:

- every method receives the same CVRP instance;
- every method receives the same deterministic initial solution;
- every method receives the same iteration budget;
- every method receives the same comparison seed;
- feasibility is audited after every candidate operator application.

The default research configuration uses multiple Q-learning training seeds and evaluates the baselines on the corresponding paired comparison seeds. The OOD split uses larger customer counts than training.

Reported fields include:

- final best objective;
- percentage improvement over the common initial solution;
- exact optimality gap when an exact certificate is available;
- best-so-far trajectory by iteration;
- iteration-to-best;
- wall-clock time-to-best;
- total search runtime;
- operator-usage counts;
- feasibility rate;
- paired Q-learning-minus-baseline objective differences with bootstrap confidence intervals.

Wall-clock values are hardware-dependent. Search-iteration counts and operator usage should be interpreted alongside runtime.

## Exactness / Verification

Small instances use an independent exact CVRP oracle:

1. a Held-Karp-style subset dynamic program computes the optimal depot tour for each capacity-feasible customer subset;
2. a set-partition dynamic program combines feasible routes to cover every customer exactly once.

The tests cross-check the exact oracle against brute-force TSP enumeration when all customers fit on one vehicle and against a hand-checkable capacity-forced partition case.

The words `optimal` or `optimality gap` are used only when this oracle returns a certificate. Larger OOD instances are evaluated without an optimality claim.

## Feasibility Audit

Every solution is checked independently for:

- missing customers;
- duplicate customers;
- invalid customer indices;
- empty routes;
- route-capacity violations;
- objective recomputation from coordinates.

The benchmark raises an error if a low-level operator produces an infeasible candidate. Infeasible routing is never counted as an improvement.

## Statistical Discipline

The unit of paired comparison is `(instance, comparison_seed)`. Q-learning results are retained across all declared training seeds; there is no best-seed substitution.

Summary statistics include mean, standard deviation, median, p90, and paired bootstrap confidence intervals for Q-learning-minus-baseline objective differences.

The CI smoke configuration is intentionally tiny and is only an integration check. It is not scientific benchmark evidence.

## Computational Cost

The benchmark records search runtime and time-to-best for every run, plus per-operator usage. Q-learning training cost occurs before deployment evaluation and is not hidden inside solve-time runtime.

At deployment, the RL selector uses one state lookup/action selection per search iteration. The first version intentionally avoids a deep network so the control overhead remains visible and small.

## Reproducibility

Randomness is explicit in JSON configuration files:

- independent data-split seed ranges;
- declared Q-learning training seeds;
- identical paired comparison seeds across methods;
- deterministic initial construction;
- deterministic stream derivation for operator and acceptance randomness.

Install:

```bash
python -m pip install -e '.[dev]'
```

Run tests:

```bash
pytest
```

Run the CI-sized smoke benchmark:

```bash
rlhh-cvrp --config configs/smoke.json --output results/smoke
```

Run the larger research configuration:

```bash
rlhh-cvrp --config configs/benchmark.json --output results/benchmark
```

Outputs include:

```text
summary.json       protocol, aggregate metrics, paired differences, operator usage
metrics.csv        method-level summaries
raw_results.jsonl  per-run trajectories and operator traces
```

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── configs/
│   ├── benchmark.json
│   └── smoke.json
├── scripts/
│   └── run_benchmark.py
├── src/rlhh_cvrp/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── exact.py
│   ├── operators.py
│   ├── problem.py
│   ├── search.py
│   └── selectors.py
├── tests/
│   ├── test_benchmark.py
│   ├── test_exact.py
│   ├── test_operators.py
│   ├── test_problem.py
│   ├── test_search.py
│   └── test_selectors.py
├── README.md
└── pyproject.toml
```

The parent `vehicle-routing-optimization` repository license applies to this project.

## Tests

The regression suite verifies methodological behavior, not imports only:

- deterministic instance generation;
- deterministic common initial construction;
- duplicate/missing-customer detection;
- independent feasibility preservation for every operator;
- deterministic operator behavior under fixed RNG seeds;
- exact-oracle agreement with independent brute-force checks;
- Q-learning update semantics and frozen-policy behavior;
- deterministic search under a fixed seed;
- complete smoke benchmark output schema;
- paired baseline/RL evaluation records;
- exact certificates on configured small smoke-test instances.

The root GitHub Actions workflow runs this project on Python 3.11 and 3.12 with dependency checking, Ruff linting, Ruff formatting checks, tests, and an end-to-end smoke benchmark.

## Experimental Interpretation

Several outcomes are valid:

- RL beats random but not adaptive weights: state-dependent learning adds value over random choice but not over a simple ALNS mechanism.
- RL matches adaptive weights: the additional learned state abstraction may not justify its training cost.
- RL is worse: the discretized state representation or training distribution may be inadequate.
- RL improves ID but degrades OOD: learned operator preferences may be size/distribution specific.
- one fixed operator remains strongest: operator-selection complexity did not earn its place under the configured budget.

Negative outcomes are retained.

## Difference from Direct Neural Construction

A neural combinatorial optimizer typically maps instance information directly to customer/node choices and constructs a route. This project does not do that.

The policy sees a search summary and selects a **classical operator**. Feasible route editing is performed by the operator implementation, not by the policy. The policy is therefore a search controller, not a solution constructor.

## Difference from the Existing FJSP Hyper-Heuristic

The portfolio already contains an FJSP PPO hyper-heuristic in `industry-4.0-lab`. That controller chooses among feasible **dispatching rules** at scheduling decision epochs.

This project instead controls **iterative improvement search** on a complete incumbent CVRP solution. Operators edit a routing solution repeatedly, and the selector learns when to use local-improvement, cross-route, and diversification moves.

The two projects therefore share the hyper-heuristic abstraction while studying different OR control loops.

## Limitations

- Instances are synthetic Euclidean CVRP, not industrial routing data.
- The first RL selector is tabular Q-learning, not a graph/deep RL model.
- The operator portfolio is deliberately small compared with research-scale ALNS implementations.
- Search budgets are iteration-based; equal iteration count does not imply identical CPU cost per operator.
- The exact oracle is intentionally restricted to small instances.
- The OOD test changes problem size but does not cover all forms of routing distribution shift.
- Simulated-annealing parameters are fixed rather than jointly tuned with each selector.
- No claim is made that Q-learning is the best RL method for operator selection.

## Claims Boundary

This repository supports only narrow claims about the implemented protocol and observed runs.

It does **not** claim:

- state-of-the-art CVRP performance;
- superiority over mature routing solvers or research ALNS systems;
- that an RL-selected solution is optimal without an exact certificate;
- paper-level reproduction of GRLOS or any other published system;
- universal superiority of RL over adaptive weights;
- production readiness or industrial savings;
- that CI smoke results constitute scientific evidence.

## Research Context and Related Repositories

- `vehicle-routing-optimization` — classical CVRP ALNS with destroy/repair adaptive weights; the present project isolates learned search-operator selection.
- `industry-4.0-lab/projects/dynamic-manufacturing-digital-twin-rl` — PPO selects FJSP dispatch operators rather than routing local-search operators.
- `neural-combinatorial-optimization` — direct learned construction/generative CO methods, explicitly contrasted with this search-controller design.
- `sequential-decision-analytics` — broader sequential-decision methodology.

The project is standalone at runtime and does not import code from these related repositories.

## Literature Positioning

**Classical adaptive search.** Ropke and Pisinger (2006) established the adaptive large-neighborhood-search pattern in routing: use multiple competing subheuristics and update their selection based on search performance.

**Operator design in routing.** Voigt (2025) reviews 211 ALNS-for-VRP papers and documents the importance and diversity of removal/insertion operators. This motivates treating the operator portfolio as a substantive experimental object rather than an incidental implementation detail.

**Learning the adaptive layer.** Johnn, Darvariu, Handl, and Kalcsics (2024) formulate operator selection as an MDP and propose graph reinforcement learning for ALNS. Rodríguez-Esparza et al. (2024) combine reinforcement learning and adaptive simulated annealing for capacitated electric vehicle routing. These works motivate the research question but are not copied or reproduced here.

## References

1. Ropke, S., & Pisinger, D. (2006). *An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows*. Transportation Science, 40(4), 455–472. https://doi.org/10.1287/trsc.1050.0135
2. Johnn, S.-N., Darvariu, V.-A., Handl, J., & Kalcsics, J. (2024). *A Graph Reinforcement Learning Framework for Neural Adaptive Large Neighbourhood Search*. Computers & Operations Research, 172, 106791. https://doi.org/10.1016/j.cor.2024.106791
3. Rodríguez-Esparza, E., Masegosa, A. D., Oliva, D., & Onieva, E. (2024). *A new Hyper-heuristic based on Adaptive Simulated Annealing and Reinforcement Learning for the Capacitated Electric Vehicle Routing Problem*. Expert Systems with Applications, 252, 124197. https://doi.org/10.1016/j.eswa.2024.124197
4. Voigt, S. (2025). *A review and ranking of operators in adaptive large neighborhood search for vehicle routing problems*. European Journal of Operational Research, 322(2), 357–375. https://doi.org/10.1016/j.ejor.2024.05.033
