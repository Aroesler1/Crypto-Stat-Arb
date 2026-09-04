# Signed-graph clustering: literature review and implementation shortlist

Compiled 2026-09-04. Scope: methods for partitioning a signed graph whose edges
carry positive and negative weights, evaluated specifically for this repo's
pipeline (signed k-NN correlation graph over token residual returns, cluster,
trade within-cluster mean reversion). Emphasis on work published 2023-2026.

Every claim about a package (version, license, release date, maintenance status)
was checked against PyPI JSON metadata or the GitHub API on 2026-09-04. Claims
that could not be checked are collected in the "Cannot verify" section rather
than guessed at.

## 0. Constraints this review is written against

### 0.1 The interface a new method must match

`stat_arb/clustering/sponge.py` defines the shape every clustering class in this
repo follows:

```python
class SPONGEClustering:
    def __init__(self, n_clusters=3, tau_plus=1.0, tau_minus=1.0, random_state=42): ...
    def fit(self, adjacency, n_clusters=None) -> "SPONGEClustering": ...
    def fit_predict(self, adjacency, n_clusters=None, symmetric=False) -> np.ndarray: ...
```

Contract, as implemented in `sponge.py`, `bnc.py` and `signed_spectral.py`:

- `adjacency` is a single signed matrix, `np.ndarray` or `pd.DataFrame`. Every
  implementation calls `adjacency.values` if it is a DataFrame, then splits
  internally with `A_plus = np.clip(A, 0, None)`, `A_minus = np.clip(-A, 0, None)`.
  There is no separate `(W_pos, W_neg)` entry point at the clustering layer,
  even though `stat_arb/graphs/signed_graph.py` can produce that pair.
- `n_clusters` is always supplied, either at construction or per-call. No
  existing class infers k. `KSelector` in `k_selection.py` is a separate object
  that consumes an eigenvalue spectrum or an embedding.
- After `fit`, the object exposes `labels_` (int array of length n),
  `eigenvalues_`, `eigenvectors_` and `embedding_` (the row-normalized
  n-by-k matrix that k-means was run on). A non-spectral method should set
  `eigenvalues_` and `embedding_` to `None` rather than fabricate them, and
  `KSelector.eigengap` must not be called on it.
- The final partition comes from `sklearn.cluster.KMeans(n_init=10,
  random_state=self.random_state)`. `random_state` is load-bearing: the repo has
  already been burned once by nondeterminism in the clustering path (see
  `handoff.md`, the `PYTHONHASHSEED` column-order defect), so any new method
  must be seeded and must be deterministic given a fixed input matrix.
- Optional but conventional: `compute_*_objective(adjacency, labels)` and
  `get_cluster_members(tokens)`.
- `stat_arb/clustering/baselines.py` (added 2026-09-04, commit 04aad1c) shows
  the accepted shape for non-spectral members of the family:
  `fit` / `fit_predict` / `labels_` only, plus method-specific outputs such as
  `n_clusters_found_` and `disagreements_`, and no `eigenvalues_` or
  `embedding_` attributes at all. Follow that pattern rather than setting the
  spectral attributes to `None`.

### 0.2 The graph this repo actually produces

From `stat_arb/graphs/knn_graph.py` and `signed_graph.py`: a Pearson correlation
matrix of residual returns is sparsified to a symmetric k-NN graph (default
k = 10, symmetrized by averaging the two directed selections, zero diagonal),
and the retained entries keep their signed correlation value. So:

- n is in the low hundreds. `handoff.md` reports 134 members in the baseline
  and 224-284 across universes, and the point-in-time universe tops out at 933
  tokens after exclusions. Treat n <= 1000 as the design point.
- Edge count is O(kn), roughly 10n to 20n after symmetrization, so the graph is
  sparse in the SSBM sense even though the underlying correlation matrix is
  dense. This is exactly the regime where the regularized variants of Section 2
  were designed to help.
- There are no node features. Anything that requires an X matrix must either be
  rejected or come with an explicit, stated feature-construction rule.
- Weights are real-valued in [-1, 1], not +/-1. Methods defined only for
  unweighted signed graphs (most of the correlation-clustering approximation
  literature, GraphC) need a thresholding or weighting decision before they can
  be applied, and that decision is itself a modeling choice.

### 0.3 Dependency policy

Current runtime dependencies, per `README.md`: numpy, pandas, scipy,
scikit-learn, matplotlib, statsmodels, pyarrow, pytest. The venv is Python
3.13.11. The repo is MIT licensed.

`lightgbm` and `zss` are not approved dependencies. Any recommendation that
requires a heavy new dependency, in particular `torch` and `torch_geometric`,
is flagged inline and needs the repo owner's sign-off before implementation.
Copyleft dependencies (GPL) are also flagged, because the repo ships MIT.

### 0.4 Two defects in the incumbent implementation, found while reading it

These are not literature findings, but they change what "beat SPONGE" means.

1. `SPONGEClustering.fit_symmetric` is documented as SPONGEsym but does not
   compute SPONGEsym. It builds the same unnormalized `M1 = L+ + tau_minus*D-`
   and `M2 = L- + tau_plus*D+` as `fit`, then whitens with `M2^{-1/2}`. That is
   an algebraically equivalent route to the same generalized eigenproblem as
   `fit`, not the normalized operator. Real SPONGEsym uses the pencil
   `(L+_sym + tau_minus*I, L-_sym + tau_plus*I)` with
   `L±_sym = I - (D±)^{-1/2} A± (D±)^{-1/2}`. Confirmed against the reference
   implementation in `signet/cluster.py`
   (https://github.com/alan-turing-institute/SigNet/blob/master/signet/cluster.py,
   methods `SPONGE` and `SPONGE_sym`), and against Cucuringu et al. 2021 Remark 1.
   The two differ materially on heterogeneous degree distributions, which a
   k-NN correlation graph has after symmetrization.
2. Embedding dimension. The repo takes k eigenvectors for every method. The
   reference implementation defaults to `eigens = k - 1` for SPONGE and
   SPONGEsym, and Cucuringu et al. 2021 argue for k-1 eigenvectors of the
   symmetric Signed Laplacian on SSBM grounds ("we will consider using only the
   k-1 smallest eigenvectors of Lsym"). Mercado et al. 2019 make the same k-1
   choice for p >= 1. This is a one-line change with a real chance of moving
   cluster quality, and it should be tested before any new method is added,
   otherwise every comparison below is against a slightly mis-specified baseline.

Also worth noting: `signet`'s SPONGE scales the eigenvector matrix by the
inverse eigenvalues (`v = v / w`) before k-means; the repo instead row-normalizes.
These are different embeddings. Neither is obviously wrong, but the difference
should be a tested switch rather than an accident.

---

## 1. SPONGE and SPONGEsym (incumbent baseline)

Cucuringu, Davies, Glielmo, Tyagi, "SPONGE: A generalized eigenproblem for
clustering signed networks", AISTATS 2019.
https://arxiv.org/abs/1904.08575

**1. What it does.** Motivated by structural balance: put positive edges inside
clusters and negative edges between them. With `A± ` the positive and negative
parts, `D±` their degree matrices and `L± = D± - A±`, SPONGE solves the
generalized eigenproblem

```
(L+ + tau_minus * D-) v = lambda (L- + tau_plus * D+) v
```

for the k (or k-1) smallest generalized eigenvectors, then runs k-means++ on the
rows. The objective is a ratio: the numerator penalizes positive edges cut
between clusters plus a `tau_minus`-weighted negative-degree term, the
denominator rewards negative edges cut. SPONGEsym replaces the unnormalized
Laplacians with `L±_sym = I - (D±)^{-1/2} A± (D±)^{-1/2}` and the degree terms
with `tau I`, giving the pencil `(L+_sym + tau_minus*I, L-_sym + tau_plus*I)`.
Equivalently, the smallest eigenvectors of
`T = (L-_sym + tau_plus*I)^{-1/2} (L+_sym + tau_minus*I) (L-_sym + tau_plus*I)^{-1/2}`,
with `G_k(T) = (L-_sym + tau_plus*I)^{-1/2} V_k(T)` recovering the generalized
eigenvectors. `tau_plus = tau_minus = 1` is the reference default.

**2. Implementation.** Already in this repo. The reference Python implementation
is `signet` (https://github.com/alan-turing-institute/SigNet). Status as of
2026-09-04, via the GitHub API: **archived on 2025-11-04, read-only**, last
push 2021-03-11, 41 stars, **no LICENSE file in the repository root** (contents
are `.gitattributes`, `.gitignore`, `README.md`, `docs`, `requirements.txt`,
`setup.py`, `signet`), and `requirements.txt` contains the single character `.`.
Not on PyPI. Practical consequence: `signet` is usable as a reference to read,
but should not be vendored or depended on, because there is no grant of license.

**3. Cost.** Dense `scipy.linalg.eigh` on a symmetric-definite pair is O(n^3).
At n = 300 that is milliseconds; at n = 1000, tens to hundreds of milliseconds.
Fine. For larger n, LOBPCG on the sparse pencil, which is what `signet` uses.

**4. Features.** Adjacency only.

**5. k.** Must be supplied.

**6. Recommendation.** Keep as the baseline. Fix `fit_symmetric` to be the real
normalized operator and add a `n_eigen` switch defaulting to k-1.

---

## 2. Regularized SPONGEsym and regularized Signed Laplacian

Cucuringu, Singh, Sulem, Tyagi, "Regularized spectral methods for clustering
signed networks", JMLR 22(264):1-79, 2021.
https://www.jmlr.org/papers/v22/20-1289.html and
https://arxiv.org/abs/2011.01737

**1. What it does.** Two contributions. Theoretically, it extends the SSBM
analysis of SPONGE to k >= 2 unequal-sized clusters. Practically, it adds
regularization so that spectral methods do not fall apart on sparse or
disconnected signed graphs. Quoting the construction: for regularization
parameters `gamma_plus, gamma_minus >= 0`, define regularized adjacencies

```
A±_{gamma±} = A± + (gamma± / n) * 1 1^T
```

that is, add a constant weight `gamma± / n` to every pair including self-loops.
With `D±_{gamma±}` their degree matrices,

```
L±_{sym,gamma±} = I - (D±_{gamma±})^{-1/2} A±_{gamma±} (D±_{gamma±})^{-1/2}
```

Regularized SPONGEsym then takes the k smallest generalized eigenvectors of the
pencil `(L+_{sym,gamma+} + tau_minus*I, L-_{sym,gamma-} + tau_plus*I)`, that is
of

```
T_{gamma+,gamma-} = (L-_{sym,gamma-} + tau_plus*I)^{-1/2}
                    (L+_{sym,gamma+} + tau_minus*I)
                    (L-_{sym,gamma-} + tau_plus*I)^{-1/2}
```

The regularized Signed Laplacian is simpler and is the cheaper of the two to
implement. With `gamma = gamma_plus + gamma_minus`,

```
A_gamma = A + ((gamma_plus - gamma_minus) / n) * 1 1^T
D_gamma = D + gamma * I
L_gamma = I - D_gamma^{-1/2} A_gamma D_gamma^{-1/2}
```

and one clusters the rows of the **k-1** smallest eigenvectors of `L_gamma`.
Note the special case `gamma_plus = gamma_minus`, where the rank-one term
vanishes and the whole method collapses to a degree-shifted normalized signed
Laplacian, `L_gamma = I - D_gamma^{-1/2} A D_gamma^{-1/2}` with
`D_gamma = D + gamma I`. That is a two-line change to `signed_spectral.py`.

Parameter guidance from the paper (Remark 15): estimate the density
`p_hat = 2 / (n(n-1)) * sum_{i<j} |A_ij|`, then set `gamma = (p_hat (n-1))^{7/8}`.
The paper explicitly says it does not know how to split that into `gamma_plus`
and `gamma_minus`. For `tau_plus, tau_minus` the paper does a grid search rather
than proposing a closed form, and reports that the ARI surface is steepest when
clusters are unbalanced and density is low, which describes a crypto k-NN graph
well.

**2. Implementation.** No maintained package. `signet` has
`spectral_cluster_adjacency_reg`, but see the licensing problem in Section 1.
The equations above are complete enough to implement directly in numpy/scipy;
that is the recommended route and adds no dependency.

**3. Cost.** Same O(n^3) dense eigendecomposition as SPONGE. The rank-one
regularizer destroys sparsity if formed explicitly, but at n <= 1000 dense is
fine anyway, and for a sparse solver it can be applied as a matrix-vector
operator (`A x + (c/n) (1^T x) 1`) without materializing it.

**4. Features.** Adjacency only.

**5. k.** Must be supplied.

**6. Recommendation.** **Implement.** Highest benefit-per-unit-effort item in
this review. It is a small delta on code that already exists, it is the
published successor to the method the repo already uses, it targets exactly the
sparse regime a k-NN graph sits in, and it has a principled default for the one
new parameter.

---

## 3. Signed Power Mean Laplacian

Mercado, Tudisco, Hein, "Spectral Clustering of Signed Graphs via Matrix Power
Means", ICML 2019, PMLR 97.
https://arxiv.org/abs/1905.06230 and
http://proceedings.mlr.press/v97/mercado19a/mercado19a.pdf

Predecessor: Mercado, Tudisco, Hein, "Clustering Signed Networks with the
Geometric Mean of Laplacians", NeurIPS 2016, https://arxiv.org/abs/1701.00757

**1. What it does.** A one-parameter family that interpolates between the known
signed Laplacians. Let

```
L+_sym = (D+)^{-1/2} L+ (D+)^{-1/2}       normalized Laplacian of the positive part
Q-_sym = (D-)^{-1/2} Q- (D-)^{-1/2}       normalized SIGNLESS Laplacian of the negative part,
                                          Q- = D- + A-
```

The scalar power mean is `m_p(a,b) = ((a^p + b^p)/2)^{1/p}`. Its matrix version
for symmetric positive definite A, B is

```
M_p(A, B) = ((A^p + B^p) / 2)^{1/p}
```

where `Y^{1/p}` is the unique positive definite solution of `X^p = Y`. The
Signed Power Mean Laplacian is

```
L_p = M_p(L+_sym, Q-_sym)
```

For `p < 0` the arguments must be positive definite, so use the shifted pair
`L+_sym + eps*I` and `Q-_sym + eps*I` for some `eps > 0`. Algorithm 1 in the
paper: set `k' = k-1` if `p >= 1` else `k' = k`, take the `k'` smallest
eigenvectors of `L_p`, k-means the rows.

Special cases: `p = 1` recovers the arithmetic-mean Laplacian
`L_AM = L+_sym + Q-_sym` (up to a factor of 2 this is the Kunegis signed
Laplacian family); `p -> 0` recovers the geometric mean
`L_GM = L+_sym # Q-_sym` with `A # B = A^{1/2}(A^{-1/2} B A^{-1/2})^{1/2} A^{1/2}`;
`p = -1` is the harmonic mean. The paper's SBM analysis is the interesting part:
it shows the arithmetic and geometric means, which were the prior art, are
suboptimal, and that smaller p is more permissive about recovering clusters that
are only informative in one of the two layers. Since `m_p` is monotone in p, the
family is ordered, and p behaves like a "how much do I trust the weaker signal"
dial. For a correlation graph where the negative-correlation layer is much
sparser and noisier than the positive layer, this is a directly relevant knob.

**2. Implementation.** Reference code is `https://github.com/melopeo/SPM`.
Checked 2026-09-04: **MATLAB**, 7 stars, last push 2019-06-07, **no license
file**. There is no Python package. Implementation must be from the equations.
For n <= 1000, `M_p` can be computed with a dense symmetric eigendecomposition
of each argument (raise eigenvalues to the p-th power, average, take the 1/p
power), so the "scalable Krylov subspace" machinery the paper adapts from
Mercado et al. 2018 is not needed here.

**3. Cost.** Naive dense route: two O(n^3) eigendecompositions to form `L_p`,
then one more to diagonalize it. Still under a second at n = 1000. For large
sparse graphs the paper uses a Krylov method that never forms `L_p`, which is
where the real engineering cost would be, and this repo does not need it.

**4. Features.** Adjacency only.

**5. k.** Must be supplied. p must also be chosen; the paper sweeps it.

**6. Recommendation.** **Implement, second priority.** It is genuinely a
different operator from SPONGE, not a re-parameterization, it is cheap at this
n, and the p sweep is a natural cross-validation axis alongside the existing
k sweep. Risk: one more hyperparameter to overfit, so p must be selected on the
training fold, not on backtest Sharpe.

---

## 4. Balance Normalized Cut and Balance Ratio Cut

Chiang, Whang, Dhillon, "Scalable clustering of signed networks using balance
normalized cut", CIKM 2012, pp. 615-624.
https://dl.acm.org/doi/10.1145/2396761.2396841

**1. What it does.** Identifies the weakness of the k > 2 signed Laplacian
objective and replaces it with a balanced cut based on the operator
`L_BNC = Dbar^{-1/2} (D+ - A) Dbar^{-1/2}`, where `A = A+ - A-` and
`Dbar = D+ + D-`. Take the k smallest eigenvectors, k-means the rows.
Equivalently, since `(D+ - A) + (A + D-) = Dbar`, maximize
`Dbar^{-1/2} (A + D-) Dbar^{-1/2}` and take its **largest** eigenvectors. The
`signet` reference (`spectral_cluster_bnc`) uses the second form, calling
`eigsh(..., which='LA')` on `Dbar^{-1/2}(A + D_n)Dbar^{-1/2}` with
`D_n = D-`, then scaling `v = v * w`.

**2. Implementation.** `bnc.py` in this repo, but it does not match the paper.
It solves `eigh(A+ - A-, D_tot)` and takes columns `[:, :k]`, and `scipy.eigh`
returns eigenvalues in ascending order, so it is selecting the **most negative**
generalized eigenvalues of the pencil `(A, Dbar)`. Two problems: the `+ D-` term
of the BNC operator is missing entirely, and the wrong end of the spectrum is
taken (BNC wants the largest of `Dbar^{-1/2}(A + D-)Dbar^{-1/2}`, equivalently
the smallest of `I` minus that). Reconcile against
https://github.com/alan-turing-institute/SigNet/blob/master/signet/cluster.py
before treating `BNCClustering` as a published-method comparison point.

**3-5.** O(n^3) dense; adjacency only; k supplied.

**6. Recommendation.** Already present. Verify against the paper rather than
adding anything new.

---

## 5. Signed Bethe-Hessian

Saade, Krzakala, Zdeborova, "Spectral Clustering of Graphs with the Bethe
Hessian", NeurIPS 2014.
https://arxiv.org/abs/1406.1880

Stephan, Zhu, "Community detection with the Bethe-Hessian", COLT 2025.
https://arxiv.org/abs/2411.02835

**1. What it does.** The Bethe-Hessian, also called the deformed Laplacian, is

```
H(r) = (r^2 - 1) I - r A + D
```

with r a real parameter tied to the expected degree (the canonical choice is
`r = sqrt(mean degree)`). Its informative eigenvectors are those with negative
eigenvalues. It matches the detection threshold of the non-backtracking operator
while staying real symmetric, so it is the right tool when the graph is sparse
enough that the ordinary Laplacian's spectrum is swamped by degree fluctuations.
Stephan and Zhu 2025 give the first rigorous proof that **the number of negative
outliers of H(r) consistently estimates the number of blocks** above the
Kesten-Stigum threshold when the expected degree is at least 2. That makes it a
k-selection tool as well as a clustering tool.

**2. Implementation.** `signet` has `spectral_cluster_bethe_hessian` (unusable
for licensing reasons, see Section 1). No maintained Python package. The
operator is three lines of numpy.

**3. Cost.** O(n^3) dense, or a few Lanczos iterations for the negative end of
the spectrum. Trivial at this n.

**4. Features.** Adjacency only.

**5. k.** This is the one method in this review that can genuinely **infer k**,
by counting negative eigenvalues, and it comes with a 2025 consistency proof for
that specific procedure.

**6. Recommendation.** **Implement as a k-selector first, clustering second.**
The clear caveat: Stephan and Zhu 2025 analyze the *unsigned* SBM. The signed
extension is in Saade, Krzakala, Lelarge, Zdeborova, "Clustering from sparse
pairwise measurements" (ISIT 2016), but I could not open a full text to confirm
the exact signed operator and its r calibration, so that is listed under
"Cannot verify". Implement the unsigned-form counter as a diagnostic against
`KSelector.eigengap`, and do not claim the consistency result carries over to
signed weighted graphs without checking the ISIT paper.

---

## 6. Correlation clustering and Pivot

Ailon, Charikar, Newman, "Aggregating inconsistent information: ranking and
clustering", J. ACM 55(5):23:1-23:27, 2008.
https://dl.acm.org/doi/10.1145/1411509.1411513
Preprint: http://dimacs.rutgers.edu/~alantha/papers2/aggregating_journal.pdf

**1. What it does.** Correlation clustering minimizes disagreements: positive
edges that end up between clusters plus negative edges that end up within
clusters. That is exactly the objective `SPONGEClustering.compute_sponge_objective`
already computes, which makes Pivot a directly comparable baseline on the repo's
own metric. Pivot is three lines:

```
while there are unclustered nodes:
    pick a uniformly random unclustered pivot node u
    form the cluster {u} union {v unclustered : A_uv > 0}
    remove that cluster from the graph
```

On a complete signed graph this is a randomized 3-approximation for min
disagreements. In practice one runs it many times with different random
permutations and keeps the best objective value.

Modern context, for calibration of how much is left on the table:

- Cohen-Addad, Lee, Newman, "Correlation Clustering with Sherali-Adams",
  FOCS 2022, https://arxiv.org/abs/2207.10889, gives (1.994 + eps).
- Cao, Cohen-Addad, Lee, Li, Newman, Vogl, "Understanding the Cluster LP for
  Correlation Clustering", STOC 2024, https://arxiv.org/abs/2404.17509. Important
  correction: the v3 revision of 2025-10-31 states that the conference version's
  1.437 ratio has a gap in its proof, and the currently provable bound is
  **(1.485 + eps)**. Cite 1.485, not 1.437.
- Dalirrooyfard, Makarychev, Mitrovic, "Pruned Pivot", ICML 2024,
  https://arxiv.org/abs/2402.15668, and "Correlation Clustering Beyond the Pivot
  Algorithm", ICML 2025, https://arxiv.org/abs/2404.06797, are dynamic and
  parallel variants. Neither is relevant at n <= 1000.

These LP and SDP rounding algorithms are asymptotic-guarantee objects, not
practical code at this scale; the practical member of the family is Pivot.

**2. Implementation.** No maintained PyPI package found. Two unmaintained
GitHub implementations exist (`Garrafao/correlation_clustering`,
`ajc-07/correlation_clustering`). Pivot itself is roughly 20 lines and should
simply be written in-repo.

A serious engineered alternative: `KaHIP/ScalableCorrelationClustering`
(https://github.com/KaHIP/ScalableCorrelationClustering), **MIT licensed**, last
push 2026-04-15, implementing Hausberger, Faraj, Schulz, "Scalable Multilevel and
Memetic Signed Graph Clustering", ALENEX 2025 (https://arxiv.org/abs/2208.13618).
It provides `scc` (multilevel label propagation plus FM local search) and
`scc_evolutionary` (MPI memetic). It is C++ with CMake and MPI, reads METIS
format with signed edge weights, and has no first-party Python binding. That is a
real build dependency for a repo that currently pip-installs eight pure-Python
packages, and its value at n = 300 is close to zero, since multilevel
coarsening is designed for graphs many orders of magnitude larger.

**3. Cost.** Pivot is O(n^2) per run for a dense signed matrix, O(n + m) for a
sparse one, times the number of restarts. Negligible.

**4. Features.** Adjacency only. Weighted correlation clustering works on
weighted signed edges directly if you weight disagreements by |A_ij|; the
3-approximation guarantee is for the unweighted complete-graph case, so on a
weighted k-NN graph treat Pivot as a heuristic, not an approximation algorithm.

**5. k.** **Inferred.** Pivot returns however many clusters fall out. This is
its distinctive property in this review and the reason to keep it: it is an
independent read on how many clusters the graph actually supports, free of the
eigengap heuristic.

**6. Recommendation.** **Use as a baseline. Already done.** A concurrent session
added `stat_arb/clustering/baselines.py` (commit 04aad1c, 2026-09-04) containing
`PivotCorrelationClustering` with weighted disagreements, `n_restarts` best-of
selection, an `n_clusters_found_` output and an optional `match_k` merge that is
off by default. That matches the recommendation in this section, so no further
work is needed here. The same file adds `SignedHierarchicalClustering`
(average linkage on a signed correlation distance) and `PCALoadingKMeans`, which
are sensible non-spectral controls but are not from this literature. Skip the
KaHIP build.

---

## 7. Signed stochastic block model inference

The SSBM is the generative model under which SPONGE, the regularized methods and
the power-mean Laplacian are all analyzed (see the model definitions in
https://arxiv.org/abs/2011.01737 Section 2.3 and https://arxiv.org/abs/1905.06230
Section 3). Fitting it directly, rather than using it only for analysis, is a
different approach.

**1. What it does.** Posit that node i has latent label z_i, and that the sign
and presence of edge (i,j) depend only on (z_i, z_j) through a block matrix.
Fit by maximum likelihood, variational EM, or pseudo-likelihood, and select the
number of blocks by a penalized criterion (ICL, minimum description length,
integrated likelihood). Recent entries:

- Chen, Tang, Zhu, "Balanced Stochastic Block Model for Community Detection in
  Signed Networks", arXiv 2026-02-16, https://arxiv.org/abs/2602.14942. Bakes
  balance theory into the generative process so that balanced triangles are
  favored, and fits with a fast profile pseudo-likelihood algorithm with a
  convergence guarantee. This is the most on-point recent SBM paper for this use
  case.
- Tang, Yang, Li, Zhao, "Community detection in signed networks: A penalized
  semidefinite programming framework", Physica A 678, November 2025,
  https://doi.org/10.1016/j.physa.2025.130978.
- Jiang, "Stochastic Blockmodeling and Variational Bayes Learning for Signed
  Network Analysis", IEEE TKDE, https://ieeexplore.ieee.org/document/7917265/,
  and the exact-ICL signed SBM at https://ieeexplore.ieee.org/document/8611282/.
- Noroozi, Pensky, "Signed Diverse Multiplex Networks: Clustering and Inference",
  IEEE Trans. Inf. Theory 2024, https://arxiv.org/abs/2402.10242.

**2. Implementation.** No maintained Python package implements a *signed* SBM.
General SBM packages exist (`sparsebm` on PyPI, `pysbm` on GitHub) but are
unsigned or binary. `graph-tool` supports weighted SBMs with real-valued edge
covariates via `rec_types=["real-normal"]` and nonparametric model selection with
`minimize_nested_blockmodel_dl`
(https://graph-tool.skewed.de/static/docs/stable/demos/inference/inference.html),
which is the closest thing to an off-the-shelf signed SBM with automatic k.
Practical blocker: the `graph-tool` PyPI entry is a stale placeholder (version
2.11, **zero distribution files**), so it is not pip-installable; the supported
routes are conda, apt, or homebrew, and it is a large C++/Boost build. That does
not fit a repo whose install line is a single `pip install`.

**3. Cost.** Variational EM and pseudo-likelihood are O(n^2 k) per sweep for a
dense signed matrix, times sweeps, times restarts, times candidate k values.
Perfectly tractable at n = 1000 in numpy, but the code is a few hundred lines
with real convergence and label-switching handling.

**4. Features.** Adjacency only.

**5. k.** **Inferred**, via ICL/MDL, which is the reason to care.

**6. Recommendation.** **Skip for now, revisit if k-selection proves to be the
binding constraint.** The payoff is principled k selection, not better partitions
at fixed k, and the cost is writing an EM fitter from scratch with no reference
implementation to test against. If k-selection does become the bottleneck, the
cheaper first move is the Bethe-Hessian negative-eigenvalue count (Section 5) or
signflip parallel analysis (Section 12), not a full SBM fitter.

---

## 8. SSSNET

He, Reinert, Wang, Cucuringu, "SSSNET: Semi-Supervised Signed Network
Clustering", SDM 2022, pp. 244-252.
https://arxiv.org/abs/2110.06623 and https://doi.org/10.1137/1.9781611977172.28
Code: https://github.com/SherylHYX/SSSNET_Signed_Clustering (MIT, last push
2024-10-13, 24 stars, not archived).

**1. What it does.** A signed GNN that produces cluster probabilities `P`
end-to-end, with no separate embed-then-k-means step. The unsupervised part is a
probabilistic balanced normalized cut,

```
L_PBNC = sum_{k=1..K}  [ P(:,k)^T (D+ - A) P(:,k) ] / [ P(:,k)^T Dbar P(:,k) ]
```

where `A = A+ - A-` is the signed adjacency, `D+` the positive degree matrix and
`Dbar` the total-degree matrix, so the columns of P act as relaxed cluster
indicators. This is the Chiang et al. BNC objective made differentiable. When
seed labels exist, it adds cross-entropy plus a triplet loss
`L_triplet = (1/|T|) sum ReLU(CS(z_i, z_j) - CS(z_i, z_k) + alpha)` over cosine
similarities of embeddings.

**2. Implementation.** Also shipped inside
`torch-geometric-signed-directed`. Checked on PyPI 2026-09-04: latest **1.2.0,
released 2026-09-03**, MIT, `Requires-Python >= 3.10`, requires
`torch >= 2.4.0`, `torch_geometric >= 2.6.0`, `networkx >= 2.7`, scikit-learn,
numpy, scipy. The wheel is `py3-none-any`, so it is pure Python and has no
compiled 3.13 problem of its own; the Python-3.13 question reduces entirely to
torch and PyG, and torch ships cp313 wheels. The GitHub repo
(https://github.com/SherylHYX/pytorch_geometric_signed_directed) is **actively
maintained**: last push 2026-09-03, 148 stars, 0 open issues, MIT. Its PyPI
trove classifiers stop at Python 3.12, which is a stale classifier list rather
than evidence of incompatibility, but it does mean 3.13 is untested upstream.

So: plausibly installable on Python 3.13, at the cost of pulling in torch and
torch_geometric. **That is a very heavy new dependency for this repo** (torch and
torch_geometric are large binary distributions, hundreds of megabytes installed
even on the CPU-only macOS arm64 build) and needs the repo owner's explicit
approval. It would transform the install surface of a project whose entire
current dependency set is eight pure-Python scientific packages.

**3. Cost.** Training is O(epochs * (m * d + n * d^2)) for hidden width d. At
n = 300 that is fast in wall-clock terms, but it introduces GPU/CPU
nondeterminism, seed sensitivity, and an early-stopping decision, all of which
are hostile to a backtest that must reproduce bit-for-bit (see `handoff.md`).

**4. Features.** **Requires node features.** For featureless graphs the paper
constructs them from the graph itself: "eigenvectors corresponding to the largest
K eigenvalues" of the symmetrized adjacency for synthetic data, and
"eigenvectors corresponding to the smallest K eigenvalues of the symmetrically
normalized Signed Laplacian" for real-world data. Note what this means: on a
featureless graph SSSNET's input is a spectral embedding of the signed Laplacian.
The GNN is being asked to refine an embedding that the repo can already compute
in one `eigh` call. The expected marginal gain over spectral clustering is
therefore much smaller here than on a graph with genuine node covariates.

**5. k.** Supplied. The paper is explicit that in the fully self-supervised case
"only the number of clusters, K, is given".

**6. Recommendation.** **Skip.** Semi-supervised by design and this repo has no
labels; the featureless mode reduces to refining a spectral embedding; and it
requires torch plus torch_geometric. If the owner ever wants a GNN lane, this is
the one to pick, because it is the best-maintained option in the space and comes
from the same group as SPONGE, but it should not be the next thing built.

---

## 9. DSGC (deep signed graph clustering via weak balance)

Zhao, Li, Zhang, Wang, Zhu, Liao, "Robust Deep Signed Graph Clustering via Weak
Balance Theory", WWW 2025.
https://arxiv.org/abs/2502.05472 and https://dl.acm.org/doi/10.1145/3696410.3714915

**1. What it does.** Four stages. (a) Violation Sign-Refine: denoise edge signs
with a high-order neighborhood score `Gamma_ij = sum_l alpha_l (mu_l^+(i,j) -
mu_l^-(i,j))`, where the mu terms count positive and negative l-hop paths.
(b) Density-based augmentation: add intra-cluster positive and inter-cluster
negative edges via powers of the adjacency, following weak balance ("the enemy
of my enemy is not necessarily my friend", which permits k > 2 factions).
(c) A clustering-oriented signed GNN. (d) Assignment optimization on

```
L = (1/|V|) sum_k  Pi(:,k)^T (L+ + A-) Pi(:,k)  +  lambda * L_regu
L_regu = -(1/|V|) sum_k  Pi(:,k)^T Dbar Pi(:,k)
```

The `(L+ + A-)` operator is the interesting part and is separable from the neural
machinery: it penalizes positive edges cut and negative edges kept, and the
regularizer prevents collapse to one cluster. Weak balance is also arguably the
right theory for a correlation graph, where "not correlated with the same things"
does not imply "correlated with each other".

**2. Implementation.** https://github.com/yaoyaohuanghuang/DSGC. Checked
2026-09-04: exists, Python, **no license file**, 0 stars, last push
2024-10-18 (which predates the paper's arXiv posting). Requires torch. No PyPI
package. Not something to depend on.

**3. Cost.** GNN training plus the augmentation step, which uses matrix powers
and is O(n^3) dense or O(n * m) sparse per power.

**4. Features.** **Requires node features X.** Same fallback as SSSNET: "Node
features X are derived from the K-dimensional embeddings corresponding to the
largest K eigenvalues of the symmetrized adjacency matrix". Same critique
applies.

**5. k.** Supplied.

**6. Recommendation.** **Skip the method, borrow one idea.** Do not add torch for
this. The violation sign-refine step, however, is a pure preprocessing operator
on the adjacency matrix, needs no neural network, and is directly applicable to a
noisy correlation k-NN graph where a single spurious negative correlation can
distort a cluster. If cluster instability across rebalances turns out to be the
problem, implement sign-refine as an optional `preprocess` on the adjacency and
feed the result to the existing SPONGE.

---

## 10. GraphC (parameter-free hierarchical signed clustering)

Shebaro, Rusnak, Burtscher, Tesic, "GraphC: Parameter-free Hierarchical
Clustering of Signed Graph Networks", arXiv 2411.00249, v1 2024-10-31,
v2 2025-01-14. https://arxiv.org/abs/2411.00249

**1. What it does.** Hierarchical bisection driven by a loss that is the
correlation-clustering objective in fractional form,

```
V_ij = L_Sigma_ij = pos_out + neg_in
```

with `pos_out` the fraction of positive edges between clusters and `neg_in` the
fraction of negative edges within clusters, generalized to

```
L_Sigma_ij(alpha, beta) = beta * (alpha * pos_out + (1 - alpha) * neg_in)
                          + (1 - beta) * |V_iso| / |V|
```

where alpha weights the two error types (0.5 is symmetric) and beta penalizes
isolated vertices. The algorithm labels each connected component uniformly, then
repeatedly finds the best **Harary cut** of a component (via spanning-tree
sampling over fundamental cycles) and commits the split only if the global loss
improves by at least epsilon, otherwise it undoes the split and freezes that
component. Reported average cumulative improvement of 18.64% over the second-best
baseline on fourteen datasets.

**2. Implementation.** The paper points at an anonymous review repository. The
real one appears to be https://github.com/DataLab12/graphC: checked 2026-09-04,
**C++**, **no license file**, 1 star, last push 2025-05-13, contents are
`GraphBplus_Harary.cpp`, `index.cpp`, `2022_Baseline`, `README.md`. Not
installable, not licensed, not Python.

**3. Cost.** No asymptotic bound is stated. Per component the Harary-cut search
is DFS at O(|V| + |E|) times up to I spanning-tree samples, times the number of
splits. Empirically it is presented as scalable, but with I as a hidden constant.

**4. Features.** Adjacency only, purely structural.

**5. k.** **Inferred.** This is the selling point: parameter-free, k falls out of
the epsilon stopping rule.

**6. Recommendation.** **Skip.** The named benefit (no k) is real but is bought
with alpha, beta, epsilon, Gamma and the spanning-tree sample count, so
"parameter-free" is an overstatement. There is no usable implementation, the
objective is essentially the same one Pivot targets, and Pivot also infers k for
a fraction of the effort. Note also that GraphC's loss is defined on edge counts,
not weights, so the repo's real-valued correlations would have to be thresholded
to +/-1 first, discarding information.

---

## 11. Local search for polarized communities

Aronsson, Haghir Chehreghani, "An Efficient Local Search Approach for Polarized
Community Discovery in Signed Networks", NeurIPS 2025.
https://arxiv.org/abs/2502.02197 (v1 2025-02-04, v4 2026-03-07)

**1. What it does.** Finds k polarized communities that are internally cohesive
and externally antagonistic, while explicitly allowing **neutral vertices that
belong to no community**, and with an objective constructed to avoid the
size-imbalanced degenerate solutions that plague earlier polarity-maximization
work. Optimization is by local search connected to block-coordinate Frank-Wolfe,
with a proved linear convergence rate.

**2. Implementation.** No code URL found in the abstract or listing page; see
"Cannot verify".

**3. Cost.** Linear convergence rate proved; per-iteration cost is local search
over vertex moves, so O(m) or O(n^2) per sweep depending on representation.
Tractable at this n.

**4. Features.** Adjacency only.

**5. k.** Supplied.

**6. Recommendation.** **Skip for the main lane, note for later.** The neutral
vertex concept is the genuinely interesting property for this repo: a k-NN
correlation graph contains tokens that are simply not co-moving with anything,
and forcing them into a cluster injects noise into a within-cluster mean-reversion
book. Today the repo handles this with a noisy-cluster pick. If that mechanism
proves inadequate, this is the paper to come back to. Without released code it is
a from-scratch build of a nontrivial optimizer.

---

## 12. k-selection methods worth adding to KSelector

`k_selection.py` currently offers eigengap, silhouette, Calinski-Harabasz,
Davies-Bouldin and gap statistic. Two additions from the recent literature are
better matched to signed spectral embeddings than silhouette-style criteria:

- **Signflip parallel analysis.** Hong, Cape, "Network Signflip Parallel Analysis
  for Selecting the Embedding Dimension", IEEE Trans. Inf. Theory, accepted;
  arXiv 2509.05722 (v1 2025-09-06, revised 2026-07-28),
  https://arxiv.org/abs/2509.05722. Compare the eigenvalues of the normalized
  adjacency to those obtained after randomly signflipping its entries, and keep
  the eigenvalues that exceed that empirical noise floor. It is data-driven,
  needs no tuning, and it is a permutation-style test rather than a heuristic
  gap, so it degrades gracefully when the spectrum has no clean gap. Maybe 40
  lines: signflip the matrix B times, take the max eigenvalue each time, threshold.
  Its native setting is unsigned networks, and this repo's matrix already has
  signs, so the null construction needs a moment's thought before use.
- **Bethe-Hessian negative-eigenvalue count**, Section 5, with the caveat there.

**Recommendation.** Implement signflip parallel analysis as an extra `KSelector`
method. It is self-contained, adds no dependency, and gives a second opinion on
k that does not share the eigengap's failure modes.

---

## 13. Signed modularity and signed Leiden

Traag, Bruggeman, "Community detection in networks with positive and negative
links", Phys. Rev. E 80(3):036115, 2009. https://arxiv.org/abs/0811.2329

**1. What it does.** Extends the Potts/modularity quality function to signed
graphs by scoring the positive and negative subgraphs separately and combining
them, typically with a weight `0 <= alpha <= 1` on the positive part. Optimized
greedily by Louvain or Leiden moves.

**2. Implementation.** `leidenalg` on PyPI. Checked 2026-09-04: version
**0.12.0, released 2026-05-24**, depends on `python-igraph`,
`Requires-Python >= 3.7`. Wheels are `cp38-abi3` including
`macosx_11_0_arm64`, and abi3 wheels are forward compatible, so it **does**
install on Python 3.13 despite the absence of cp313-specific wheels. The blocker
is licensing: `leidenalg` is **GPL-3.0-or-later** and `python-igraph` is GPL-2.0
or later. This repo is MIT. A GPL runtime dependency in an MIT public research
repo is a licensing decision for the owner, not a technical one.

**3. Cost.** Near-linear in edges. Trivial at this n.

**4. Features.** Adjacency only.

**5. k.** **Inferred**, controlled indirectly by the resolution parameter.

**6. Recommendation.** **Skip, or implement signed modularity directly.** The
quality function itself is a dozen lines and can be evaluated on any candidate
partition with no dependency at all; adding it as a scoring function in
`compute_*_objective` style gives most of the diagnostic value. Pulling in a GPL
package for greedy optimization at n = 300 is not worth the license question.

---

## 14. Adjacent 2025-2026 work, noted but not recommended

- Lee et al., "SSGC: A relaxed semi-streaming framework for Scalable Signed Graph
  Clustering", Future Generation Computer Systems, 2026,
  https://doi.org/10.1016/j.future.2026.108712. Semi-streaming, aimed at graphs
  too large to hold in memory. Irrelevant at n <= 1000.
- Shin et al., "Improving the Accuracy of Community Detection on Signed Networks
  via Community Refinement and Contrastive Learning" (ReCon), WWW 2026,
  https://arxiv.org/abs/2601.16372. A model-agnostic post-processing wrapper
  (structural refinement, boundary refinement, contrastive learning, clustering)
  that sits on top of any base method. Conceptually attractive as a bolt-on to
  SPONGE, but the contrastive-learning stage implies a torch dependency and no
  code was found.
- Diaz-Diaz, Devriendt, Lambiotte, "Gremban Expansion for Signed Networks",
  arXiv 2509.14193, https://arxiv.org/abs/2509.14193. Lifts a signed graph to a
  larger unsigned one with a bijection between symmetry-respecting cut-sets in
  the lift and cut-sets plus frustration sets in the original, which lets
  unsigned machinery be reused and separates "communities" from "factions". Nice
  theory; doubles n; no released code found.
- Zhang et al., "Signed Graph Representation Learning: A Survey", arXiv
  2402.15980, 2024, https://arxiv.org/abs/2402.15980. Useful map of the
  embedding literature, but embedding-first, so most of it inherits the node
  feature problem.
- Noroozi, Pensky, "Signed Diverse Multiplex Networks: Clustering and Inference",
  IEEE Trans. Inf. Theory 2024, https://arxiv.org/abs/2402.10242. Multiplex, so
  it would apply to a stack of correlation graphs at different horizons rather
  than a single one. Possible future direction if the repo ever builds a
  multi-horizon graph.

## 15. Application evidence: signed clustering for statistical arbitrage

Two papers do essentially the pipeline this repo does, and both find SPONGE
competitive, which is useful calibration for how much upside a new clusterer can
realistically deliver.

- Jin, Cucuringu, Cartea, "Correlation Matrix Clustering for Statistical
  Arbitrage Portfolios", ICAIF 2023, https://doi.org/10.1145/3604237.3626894,
  SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4560455. Market
  residual returns (returns minus CAPM beta times market return), correlation
  matrix treated as a signed network, five clustering algorithms compared: two
  SPONGE variants, a modified spectral method and two signed Laplacian variants.
- Korniejczuk, Slepaczuk, "Statistical arbitrage in multi-pair trading strategy
  based on graph clustering algorithms in US equities market", arXiv 2406.10695,
  2024-06-15, https://arxiv.org/abs/2406.10695. Sixty-day residual-return
  correlations as edge weights; spectral, two signed Laplacian variants, two
  SPONGE variants; contrarian within-cluster book on previous winners and losers.
  Reports the SPONGE strategy beating fixed Fama-French sector clusters and
  buy-and-hold, at 12.2% annualized, Sharpe 1.1, Sortino 2.01.

Read across: in published equity work the choice among signed spectral methods
moves results less than the residualization and the trading rule do. The prior
should be that a better clusterer is worth a modest improvement, not a
transformative one, and that the biggest wins in this repo are more likely to
come from k-selection stability and cluster stability across rebalances than
from a new eigenproblem.

---

## 16. Ranked recommendation table

Effort is coding plus testing time against the existing interface, assuming
numpy/scipy only. Benefit is expected improvement over the current
`SPONGEClustering` on this repo's graph, not on benchmark signed networks.

| # | Method | Verdict | Effort | Expected benefit over SPONGE | New deps |
|---|--------|---------|--------|------------------------------|----------|
| 1 | Fix `fit_symmetric` to true SPONGEsym, add `n_eigen` (k-1 default) | implement | very low (hours) | Medium. Corrects the baseline; every other comparison depends on it | none |
| 2 | Regularized Signed Laplacian (`L_gamma`, gamma = (p_hat(n-1))^{7/8}) | implement | low (half day) | Medium. Targets the sparse regime a k-NN graph is in | none |
| 3 | Regularized SPONGEsym | implement | low-medium (1 day) | Medium. Published successor to the incumbent | none |
| 4 | Pivot correlation clustering | baseline, **already shipped** in `baselines.py` | done | Low as a clusterer, high as a check. Infers k; reference value for the disagreement objective | none |
| 5 | Signflip parallel analysis in `KSelector` | implement | low (half day) | Medium, on k-selection rather than partition quality | none |
| 6 | Signed Power Mean Laplacian `L_p` | implement | medium (1-2 days) | Medium-high. Genuinely different operator; p is a real dial | none |
| 7 | Bethe-Hessian, as k-counter then clusterer | implement | low-medium (1 day) | Medium on k, unknown on partitions (signed form unverified) | none |
| 8 | Signed modularity as a scoring function only | optional | low | Low. Diagnostic value | none |
| 9 | DSGC violation sign-refine, as adjacency preprocessing only | optional | medium | Unknown. Worth trying only if cluster instability is diagnosed | none |
| 10 | Reconcile `bnc.py` with the published BNC operator | implement | low | Low, but it is a correctness issue | none |
| 11 | Signed SBM / balanced SBM fitting | skip for now | high (1 week+) | Medium on k, low on partitions. No reference implementation | none, but graph-tool is not pip-installable |
| 12 | Local search for polarized communities (NeurIPS 2025) | skip | high | Unknown. Neutral-vertex idea is attractive; no code | none |
| 13 | SSSNET | skip | high | Low. Featureless mode refines a spectral embedding the repo already has | torch + torch_geometric, owner approval |
| 14 | DSGC (full method) | skip | high | Unknown. Unlicensed code | torch, owner approval |
| 15 | GraphC | skip | high | Unknown. Unlicensed C++, needs edge thresholding | none, but a C++ build |
| 16 | KaHIP ScalableCorrelationClustering | skip | high | ~Zero at n <= 1000 | C++/CMake/MPI |
| 17 | `leidenalg` signed modularity | skip | low | Low | GPL-3.0 vs repo MIT |
| 18 | ReCon post-processing (WWW 2026) | skip | high | Unknown; no code | torch implied |
| 19 | SSGC semi-streaming | skip | n/a | Zero at this n | n/a |

## 17. Shortlist to implement behind the sponge.py interface, by benefit/effort

In order. Each is pure numpy/scipy, adds no dependency, and fits the
`fit` / `fit_predict` / `labels_` / `eigenvalues_` / `embedding_` contract.

1. **Fix the baseline.** In `sponge.py`, make `fit_symmetric` build
   `L±_sym = I - (D±)^{-1/2} A± (D±)^{-1/2}` and solve the pencil
   `(L+_sym + tau_minus*I, L-_sym + tau_plus*I)`; add an `n_eigen` argument
   defaulting to `n_clusters - 1`; add an `embedding_scaling` switch for
   row-normalize versus eigenvalue-scale. Do this first, because every number
   below is measured against it. Regression-test that the existing labels are
   reproducible under a fixed seed before and after.
2. **`RegularizedSignedSpectralClustering`** in `signed_spectral.py` or a new
   `regularized.py`. `D_gamma = D + gamma I`,
   `A_gamma = A + ((gamma_plus - gamma_minus)/n) 11^T`,
   `L_gamma = I - D_gamma^{-1/2} A_gamma D_gamma^{-1/2}`, cluster the k-1
   smallest eigenvectors. Default `gamma` from the paper's rule,
   `gamma_plus = gamma_minus = gamma/2` (which makes the rank-one term vanish),
   with both exposed. Smallest possible diff for a published improvement.
3. ~~**`PivotCorrelationClustering`** as a non-spectral baseline.~~ **Already
   shipped** in `stat_arb/clustering/baselines.py` as of commit 04aad1c
   (2026-09-04), k-free by default with best-of-`n_restarts` on weighted
   disagreements. Nothing to do; use it as the honesty check on the whole
   spectral lane, and report `n_clusters_found_` alongside the spectral k sweep.
4. **Signflip parallel analysis** as `KSelector.signflip`. B signflip replicates,
   keep eigenvalues above the empirical max. Independent second opinion on k.
5. **`RegularizedSPONGEClustering`**, the regularized SPONGEsym pencil
   `(L+_{sym,gamma+} + tau_minus*I, L-_{sym,gamma-} + tau_plus*I)`. Subsumes
   item 2's machinery, so build it after item 2 shares the regularized-adjacency
   helper.
6. **`PowerMeanLaplacianClustering`** with parameter `p`, `p = 1` reproducing
   the arithmetic mean and `p -> 0` the geometric mean as tested special cases.
   Highest ceiling of the six, highest hyperparameter risk; select p on the
   training fold only.

Everything else in the table is skip or later. In particular, do not add torch or
torch_geometric on the strength of this review: the two GNN methods that would
need it (SSSNET, DSGC) both construct their node features from a spectral
embedding of the same adjacency matrix when no features exist, so on this repo's
data they start from where the spectral methods finish.

## 18. Cannot verify

Listed rather than guessed.

- **Signed Bethe-Hessian.** Stephan and Zhu (COLT 2025, arXiv 2411.02835)
  analyze the *unsigned* SBM; the abstract page gives no signed statement. The
  signed/censored version is attributed to Saade, Krzakala, Lelarge, Zdeborova,
  "Clustering from sparse pairwise measurements" (ISIT 2016), which I located
  only through a ResearchGate record and could not open. **The exact signed
  Bethe-Hessian operator, the calibration of r for signed graphs, and whether the
  negative-eigenvalue-counting consistency result transfers to signed graphs are
  all unverified.** Implement the unsigned form as a diagnostic only, and read
  the ISIT paper before claiming more.
- **GraphC complexity.** The paper states no asymptotic bound; my per-component
  reading (DFS at O(|V| + |E|) times I spanning-tree samples) is inferred from
  the algorithm description, not quoted.
- **Local search for polarized communities (arXiv 2502.02197).** No code URL
  found on the arXiv listing page. I did not check the NeurIPS 2025 proceedings
  supplementary material, which may contain one. Complexity beyond "linear
  convergence rate" is unverified.
- **ReCon (arXiv 2601.16372).** Could not confirm whether it needs node features,
  whether k is required, its complexity, or whether code exists. My inference
  that contrastive learning implies torch is an inference, not a confirmed fact.
- **Gremban expansion (arXiv 2509.14193).** Could not confirm whether k is
  required, whether features are needed, or whether code exists.
- **Balanced SBM (arXiv 2602.14942).** Could not confirm how the number of
  communities is selected, the complexity, or whether code is released.
- **Physica A penalized SDP (10.1016/j.physa.2025.130978).** ScienceDirect
  returned HTTP 403. Title, authors (Fengqin Tang, Han Yang, Cuixia Li, Xuejing
  Zhao), journal, volume 678, November 2025 and DOI were confirmed via Crossref;
  **the abstract, method details and any code were not retrieved.**
- **SSGC (10.1016/j.future.2026.108712).** Paywalled; confirmed only via a
  citation record and a DBLP-derived title/author list. Judged irrelevant on
  scale grounds without reading it.
- **ICAIF 2023 (10.1145/3604237.3626894).** ACM DL returned HTTP 403. Title,
  authors, venue and the list of five clustering methods came from the ORA and
  SSRN records plus search summaries, not from the paper itself. **The specific
  performance numbers per clustering method were not retrieved.**
- **DSGC repository provenance.** https://github.com/yaoyaohuanghuang/DSGC has a
  last push of 2024-10-18, which predates the paper's 2025-02-08 arXiv posting.
  I did not verify that the repository contents match the published method.
- **`torch-geometric-signed-directed` on Python 3.13.** Verified: pure-Python
  `py3-none-any` wheel, MIT, `Requires-Python >= 3.10`, 1.2.0 released
  2026-09-03. **Not verified by actually installing it**, and its trove
  classifiers stop at 3.12. The transitive `torch >= 2.4` and
  `torch_geometric >= 2.6` constraints on 3.13 for this specific machine
  (darwin/arm64) were not tested.
- **`signet` licensing.** The GitHub API reports no detected license and the
  repository root has no LICENSE file. I did not check whether a license is
  declared inside `setup.py` or the docs. Absent that, treat it as
  all-rights-reserved and do not vendor it.
- **`melopeo/SPM` and `DataLab12/graphC`** likewise have no license file
  detected. Same conclusion.

## 19. References

Primary:

- Cucuringu, Davies, Glielmo, Tyagi. SPONGE: A generalized eigenproblem for
  clustering signed networks. AISTATS 2019. https://arxiv.org/abs/1904.08575
- Cucuringu, Singh, Sulem, Tyagi. Regularized spectral methods for clustering
  signed networks. JMLR 22(264):1-79, 2021.
  https://www.jmlr.org/papers/v22/20-1289.html
- Kunegis, Schmidt, Lommatzsch, Lerner, De Luca, Albayrak. Spectral analysis of
  signed graphs for clustering, prediction and visualization. SDM 2010,
  pp. 559-570. https://doi.org/10.1137/1.9781611972801.49
- Chiang, Whang, Dhillon. Scalable clustering of signed networks using balance
  normalized cut. CIKM 2012, pp. 615-624.
  https://dl.acm.org/doi/10.1145/2396761.2396841
- Mercado, Tudisco, Hein. Spectral Clustering of Signed Graphs via Matrix Power
  Means. ICML 2019. https://arxiv.org/abs/1905.06230
- Mercado, Tudisco, Hein. Clustering Signed Networks with the Geometric Mean of
  Laplacians. NeurIPS 2016. https://arxiv.org/abs/1701.00757
- Saade, Krzakala, Zdeborova. Spectral Clustering of Graphs with the Bethe
  Hessian. NeurIPS 2014. https://arxiv.org/abs/1406.1880
- Stephan, Zhu. Community detection with the Bethe-Hessian. COLT 2025.
  https://arxiv.org/abs/2411.02835
- Ailon, Charikar, Newman. Aggregating inconsistent information: ranking and
  clustering. J. ACM 55(5), 2008. https://dl.acm.org/doi/10.1145/1411509.1411513
- Cohen-Addad, Lee, Newman. Correlation Clustering with Sherali-Adams. FOCS 2022.
  https://arxiv.org/abs/2207.10889
- Cao, Cohen-Addad, Lee, Li, Newman, Vogl. Understanding the Cluster LP for
  Correlation Clustering. STOC 2024, v3 2025. https://arxiv.org/abs/2404.17509
- Dalirrooyfard, Makarychev, Mitrovic. Pruned Pivot. ICML 2024.
  https://arxiv.org/abs/2402.15668
- He, Reinert, Wang, Cucuringu. SSSNET: Semi-Supervised Signed Network
  Clustering. SDM 2022, pp. 244-252. https://arxiv.org/abs/2110.06623
- Zhao, Li, Zhang, Wang, Zhu, Liao. Robust Deep Signed Graph Clustering via Weak
  Balance Theory. WWW 2025. https://arxiv.org/abs/2502.05472
- Shebaro, Rusnak, Burtscher, Tesic. GraphC: Parameter-free Hierarchical
  Clustering of Signed Graph Networks. arXiv 2411.00249.
  https://arxiv.org/abs/2411.00249
- Aronsson, Haghir Chehreghani. An Efficient Local Search Approach for Polarized
  Community Discovery in Signed Networks. NeurIPS 2025.
  https://arxiv.org/abs/2502.02197
- Hausberger, Faraj, Schulz. Scalable Multilevel and Memetic Signed Graph
  Clustering. ALENEX 2025. https://arxiv.org/abs/2208.13618
- Hong, Cape. Network Signflip Parallel Analysis for Selecting the Embedding
  Dimension. IEEE Trans. Inf. Theory. https://arxiv.org/abs/2509.05722
- Chen, Tang, Zhu. Balanced Stochastic Block Model for Community Detection in
  Signed Networks. arXiv 2602.14942. https://arxiv.org/abs/2602.14942
- Tang, Yang, Li, Zhao. Community detection in signed networks: A penalized
  semidefinite programming framework. Physica A 678, 2025.
  https://doi.org/10.1016/j.physa.2025.130978
- Traag, Bruggeman. Community detection in networks with positive and negative
  links. Phys. Rev. E 80:036115, 2009. https://arxiv.org/abs/0811.2329
- Shin, Kim, Lee, Lee, Kang. Improving the Accuracy of Community Detection on
  Signed Networks via Community Refinement and Contrastive Learning. WWW 2026.
  https://arxiv.org/abs/2601.16372
- Diaz-Diaz, Devriendt, Lambiotte. Gremban Expansion for Signed Networks.
  arXiv 2509.14193. https://arxiv.org/abs/2509.14193
- Zhang, Zhao, Li, Liu, Zhang, Huang, Zhu. Signed Graph Representation Learning:
  A Survey. arXiv 2402.15980. https://arxiv.org/abs/2402.15980
- Noroozi, Pensky. Signed Diverse Multiplex Networks: Clustering and Inference.
  IEEE Trans. Inf. Theory 2024. https://arxiv.org/abs/2402.10242

Application:

- Jin, Cucuringu, Cartea. Correlation Matrix Clustering for Statistical Arbitrage
  Portfolios. ICAIF 2023. https://doi.org/10.1145/3604237.3626894
- Korniejczuk, Slepaczuk. Statistical arbitrage in multi-pair trading strategy
  based on graph clustering algorithms in US equities market. arXiv 2406.10695.
  https://arxiv.org/abs/2406.10695

Software:

- SigNet. https://github.com/alan-turing-institute/SigNet (archived 2025-11-04,
  no license file, not on PyPI)
- PyTorch Geometric Signed Directed.
  https://github.com/SherylHYX/pytorch_geometric_signed_directed and
  https://pypi.org/project/torch-geometric-signed-directed/ (MIT, 1.2.0,
  2026-09-03)
- SSSNET. https://github.com/SherylHYX/SSSNET_Signed_Clustering (MIT)
- SPM (MATLAB). https://github.com/melopeo/SPM (no license file)
- DSGC. https://github.com/yaoyaohuanghuang/DSGC (no license file)
- graphC. https://github.com/DataLab12/graphC (C++, no license file)
- KaHIP ScalableCorrelationClustering.
  https://github.com/KaHIP/ScalableCorrelationClustering (MIT)
- leidenalg. https://pypi.org/project/leidenalg/ (GPL-3.0-or-later, 0.12.0,
  2026-05-24)
- graph-tool. https://graph-tool.skewed.de (PyPI entry is a stale placeholder
  with no distribution files; install via conda/apt/homebrew)
