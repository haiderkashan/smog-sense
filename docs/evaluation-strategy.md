# Evaluation Strategy

> **Audience:** whoever writes `src/smogsense/evaluation/` and the thesis chapter. **Principle:** the evaluation protocol, metrics and thresholds are fixed *before* the 2026–27 shadow-run analysis
> (`configs/evaluation.yaml`). Changing them afterwards is allowed only with a dated entry in the verification log, and results are then reported under both versions.

## 1. Evaluation questions and hypotheses

| RQ | Question |
|---|---|
| RQ1 | Does the learned system beat the *fair* physics baseline (bias-corrected CAMS) for Lahore at 24, 48 and 72 h? |
| RQ2 | How does skill degrade as the number of available stations $N$ shrinks to zero (the **Sensor-Sparsity Curve**)? |
| RQ3 | Does meta-learning add value over simple fine-tuning of a Delhi-pretrained model? |
| RQ4 | Are the intervals honest (calibrated), and do they hold up prospectively? |

**Pre-registered hypotheses** (thresholds mirror `configs/evaluation.yaml`; "CRPSS" = $1-\mathrm{CRPS}_{\rm model}/\mathrm{CRPS}_{\rm ref}$):

| ID | Statement | Tier | Window | Pass condition |
|---|---|---|---|---|
| H5 | meta vs supervised init (M7/M6) | 1, powered | Delhi season T, N∈{1,2}, ≥20 stations | CI upper bound < 1 and point ≤ 0.97 |
| H1 | HM vs M2 at 24 h | 1, indicative | Lahore confirmatory window, leave-stations-out, N_H = min(5, eligible−3) | CRPSS lower 95 % bound > 0 and point ≥ 0.15 |
| H2 | HM vs M2 at 72 h | 1, indicative | same | CRPSS lower 95 % bound > 0 and point ≥ 0.10 |
| H3 | HM vs M1, every horizon | 1, indicative | same | each horizon ≥ 0.25 with lower bound > 0 |
| H4 | sensor-free adds skill | 1, indicative | same, N=0 | ≥ 0.05 vs M1 (deterministic CRPS) |
| H6 | calibration | 2, descriptive | pooled | report coverage with CI; flag if the CI excludes [0.70, 0.90] |

Holm family {H1–H4}; H5 is its own family. Report n_eff beside every result; below ~25 the Lahore tests are indicative only.
Power illustration: "Hand-computed under a normal approximation, one-sided α=0.05, independent days, n_eff=28, with σ the assumed daily SD of log-CRPS ratio, which is unknown until a pilot. σ=0.3: H1=0.89, H2=0.59. σ=0.4: H1=0.69, H2=0.40. σ=0.5: H1=0.53, H2=0.30. Holm correction lowers these further. For a 56-day confirmatory window at ρ≈0.5, n_eff ≈ 19, lower still."

These thresholds are *hypotheses*, not promises: typical post-processing gains of a 40 km model with local data are large against raw CAMS (which under-predicts Lahore's peaks) and moderate against a bias-corrected CAMS. A failed hypothesis is a result.
Primary tests across horizons use Holm correction (§8).

## 2. Forecast targets and verification data

* **Target:** the 24-hour block mean $y_{s,T,h}$ for $h\in\{24,48,72\}$, valid iff ≥ 18 of 24 hours are valid and not imputed (US EPA's 75 % convention). Imputed hours never count.
* **Truth:** QC'd OpenAQ PM2.5. Two truth classes are always reported separately: **reference monitors** (`isMonitor`) and **low-cost sensors**. Tail behaviour (> 225 µg m⁻³) is verified on reference monitors, because optical sensors saturate and bias the upper tail low.
* **City-level verification** (what the bulletin shows) uses the mean over eligible stations that pass completeness, one value per co-located group.
* **Two truth stages:** *provisional* (OpenAQ API, window end + 6 h) for fast monitoring; *settled* (OpenAQ AWS archive, available 72 h after the end of the local day; the pipeline waits 96 h) for all reported results. Differences between the two are themselves reported.

## 3. Baselines

A claim to "beat CAMS" is only meaningful against a baseline that has *seen local data*. The ladder (full definitions in [ML architecture §10](ml-architecture.md#10-model-ladder-and-ablations)) therefore contains:

| ID | Definition | Probabilistic form |
|---|---|---|
| **M0 Persistence** | $\hat y_h=\bar x_{[T-24,\,T)}$ (the last complete 24 h block), for every $h$ | empirical quantiles of log-ratio residuals $\ln\frac{1+y}{1+\hat y}$ by horizon and season (training period) |
| **M1 Raw CAMS** | block mean of CAMS PM2.5 interpolated to the station, from the as-of run $B^\*(T)$ | same residual-quantile construction |
| **M2 CAMS-BC** | trailing 30-day regression in log space, $\ln(1+y)=a+b\ln(1+c)$ (shrunk toward $a=0,b=1$ when data are scarce; pooled across available stations); equals M1 at $N=0$ | residual quantiles from the same trailing window |
| **M3 Climatology** | quantiles of block means by domain and ±15-day day-of-year window | native |

The original plan lists only persistence and raw CAMS. Raw CAMS under-predicts Lahore\'s extremes at 40 km, so beating it is easy and says little; **M2 is the reference for H1, H2, H4.**
Baseline statistics use A only. At N=0 only M1 (deterministic) and the sensor-free learned methods are defined. Add M3b.
Deterministic baselines are scored with their exact point-forecast CRPS, $|y-\hat y|$; the probabilistic versions are scored with the same estimator as the learned models.

## 4. CRPS and quantile scoring

**Definition.** For a predictive CDF $F$ and observation $y$ (this document uses $x$ for the integration variable; the plan's $H(y-x)$ is the Heaviside function $\mathbf 1\{x\ge y\}$):

$$
\mathrm{CRPS}(F,y)=\int_{-\infty}^{\infty}\big(F(x)-\mathbf 1\{x\ge y\}\big)^2dx
=\mathbb E_F|X-y|-\tfrac12\,\mathbb E_F|X-X'| ,
$$

a strictly proper scoring rule (Matheson & Winkler, 1976; Gneiting & Raftery, 2007) in the units of the variable (µg m⁻³). For a point forecast $\hat y$ it reduces to $|y-\hat y|$. RMSE of a median is inadequate because it ignores spread and calibration.

**Quantile decomposition.** The quantile-score form is $\mathrm{CRPS}(F,y)=2\int_0^1\rho_\tau\big(y-F^{-1}(\tau)\big)\,d\tau$ (Laio & Tamea, 2007; Gneiting & Ranjan, 2011), verified numerically: for $\mathcal N(100,30^2)$ and $y=140$,
the analytic CRPS is 25.618 and $2\int\rho_\tau$ gives 25.618. This is what lets pinball-trained quantile models be scored by CRPS.

**Implementation blueprint** (`evaluation/crps.py`, `models/distribution.py`):

1. *Input:* $K=19$ non-crossing quantiles $q_1..q_K$ at $\tau_k=0.05k$ (apply monotone rearrangement first), plus the observation $y\ge0$.
2. *Reconstruct the quantile function* $Q(\tau)$: linear between knots; upper tail $q_K+s_{hi}[\Lambda(\tau)-\Lambda(\tau_K)]$, $\Lambda=-\ln(1-\tau)$; lower tail $\max\{0,\ q_1+s_{lo}[\ln\tau-\ln\tau_1]\}$ (slopes from the two outermost knots).
3. *Integrate* $2\int_{0}^{1}\rho_\tau(y-Q(\tau))d\tau$ numerically on a fine grid in $\tau$ (e.g. $4\times10^5$ points on $[10^{-6},1-10^{-6}]$ with trapezoid), vectorised over (stations × days).
4. *Point forecasts* use $|y-\hat y|$ directly (no integration).
5. *Aggregate* by arithmetic mean over valid (station, day) pairs, then report per horizon, per station class and per regime.

**Measured accuracy of the estimator** (relative error vs the exact CRPS of lognormal $\sigma\in\{0.3,0.5,0.8\}$ and Gamma(2); six observation values each):

| Levels | Mean abs. error | Max abs. error |
|---|---|---|
| $K=7$ with tail model | 1.0 – 3.5 % | 3.8 – 6.2 % |
| **$K=19$ with tail model** | **0.2 – 0.3 %** | **≤ 0.5 %** |
| $K=7$, flat tails (naïve) | – | 13.1 % |

Hence the 19-level scoring grid; the public product shows only $\tau\in\{0.10,0.50,0.90\}$. The unit tests assert < 0.5 % at $K=19$ and < 5 % at $K=7$ against analytic Normal/lognormal CRPS, and cross-check against `scoringrules` ensemble CRPS on sampled draws.

**Companion scores.**
* *Skill:* $\mathrm{CRPSS}=1-\mathrm{CRPS}/\mathrm{CRPS}_{\rm ref}$.
* *Pinball* at each $\tau$ (shows *where* in the distribution a model gains).
* *Interval score* for the 80 % interval ($\alpha=0.2$, lower $l$, upper $u$): $\mathrm{IS}=(u-l)+\frac2\alpha(l-y)\mathbf 1\{y<l\}+\frac2\alpha(y-u)\mathbf 1\{y>u\}$.
* *Threshold-weighted CRPS* (Gneiting & Ranjan, 2011) with $w(x)=\mathbf 1\{x\ge r\}$, $r=125.5$ µg m⁻³ ("very unhealthy and worse"): $\mathrm{twCRPS}=\int(F(x)-\mathbf 1\{x\ge y\})^2w(x)\,dx$ — emphasises the tail the public cares about.

## 5. Complementary metrics (calibration, events)

* **Calibration (quantile form of the PIT).** For each level, $\hat p_k=\frac1n\sum\mathbf 1\{y\le\hat q_{\tau_k}\}$; the **reliability diagram** plots $\hat p_k$ against $\tau_k$ with block-bootstrap bands. Reported: coverage of $[q_{.10},q_{.90}]$ (target 0.80 ± 0.05), $\hat p_{.50}$ (0.50 ± 0.07), and **sharpness** = mean interval width. A model that is calibrated but wide is not useful; a model that is sharp but miscalibrated is dangerous.
* **Exceedance events** at 55.5, 125.5 and 225.5 µg m⁻³: $P(y>\theta)$ comes from the quantile function; **Brier score** $\frac1n\sum(p-o)^2$ and Brier skill vs climatology; for the deterministic decision "warn if $p\ge p^\*$": POD, FAR, CSI and the **Peirce skill score** $\mathrm{POD}-\mathrm{POFD}$ (robust to rare events).
* **Category accuracy:** share of days where the median's AQI category equals the observed category, and the share within one category (an ordinal tolerance).
* **Stratified reporting** (never only pooled): by horizon; reference vs low-cost; observed-concentration regime (terciles and top 10 % of days); CAMS model cycle (before/after an upgrade); degradation mode; and $N$. RH-stratified scoring (none / 75–85 / above 85), and exceedance display rounding. RH-stratified scoring (none / 75–85 / above 85), and exceedance display rounding.

## 6. Data splits and leakage control

* **Source domain (Delhi).** Chronological: train seasons $\le S-2$, validation $S-1$, test $S$; spatial hold-out of stations; this tests pre-training in isolation.
* **Target domain (Lahore).** Blocked, with a 72-hour embargo between *adaptation* and *test* blocks: sequential 14-day adaptation windows followed by test windows; stations in the held-out set $B$ contribute **nothing** (no inputs, no labels) to adaptation and are used only as truth.
* **Purging.** Remove any sample whose 72-hour lookback or target block overlaps another split (López de Prado, 2018). Overlap is the dominant leakage path because lookbacks and targets are autocorrelated.
* **Frozen preprocessing.** Scalers, thresholds and hyper-parameters are fit/selected on source-domain training+validation data and Lahore *early-season support data only*; the pre-registered Lahore test blocks are untouched until the freeze.
* **Unit tests that make leakage a CI failure:** as-of join (no feature with `available_at > T`); target completeness; split-overlap checker; scaler provenance.
* **Multiple comparisons:** Holm correction across the three horizons for H1–H4.

* **Calendar:**
  * History: ≤ 2026-10-18 (seed baselines; E/S/V/T source splits)
  * Dev: 2026-10-19 → 11-30 (G2 probe, G3 ladder, hyper-parameters, stacker choices, mask audit/ablation, threshold pilot, replay)
  * Freeze: 2026-12-04 (freeze register: HM, thresholds, configs, bundle hashes)
  * Confirmatory: 2026-12-07 → 2027-01-31 (H1–H4, H6)
  * Delhi confirmatory: season T (H5)
  If the challenger isn't ready, freeze later; with fewer than 35 days left, H1–H4 become descriptive.

* **Calendar:**
  * History: ≤ 2026-10-18 (seed baselines; E/S/V/T source splits)
  * Dev: 2026-10-19 → 11-30 (G2 probe, G3 ladder, hyper-parameters, stacker choices, mask audit/ablation, threshold pilot, replay)
  * Freeze: 2026-12-04 (freeze register: HM, thresholds, configs, bundle hashes)
  * Confirmatory: 2026-12-07 → 2027-01-31 (H1–H4, H6)
  * Delhi confirmatory: season T (H5)
  If the challenger isn't ready, freeze later; with fewer than 35 days left, H1–H4 become descriptive.

## 7. Sensor-sparsity experiment

**Purpose.** Map forecast skill against the number $N$ of *available* Lahore stations; test the central claim that physical priors from Delhi flatten the error curve at small $N$.

**Definitions.** $N$ = number of Lahore stations contributing labels for adaptation **and** observed history at inference (see [ML architecture §7](ml-architecture.md#7-sparse-network-task-construction)). The original plan's $N=0$ ("pure zero-shot") is therefore *sensor-free mode*: no Lahore PM2.5 enters adaptation or inputs.

**Protocol** (`evaluation/sparsity_curve.py`):

1. Fix the eligible Lahore set $\mathcal S_L$ (Phase 1 audit) and $M_{\min}=3$ held-out stations; $N$-grid $\{0,1,2,3,5,8,13,21\}$ truncated at $|\mathcal S_L|-M_{\min}$.
2. For each $N$, draw $R=30$ random available sets $A$ (stratified so every station appears about equally often). For $N=0$ there is a single configuration.
3. For each $(N,A)$ and method in {M4, M5, M6, M7, M8}: adapt using only $A$'s data in the **support block**; forecast the **later test block** at held-out stations $B=\mathcal S_L\setminus A$, using only $A$ for inputs. Reference lines: M1 (constant), M2 (= M1 at $N=0$), M0, M3.
4. Score CRPS per (draw, station-day); aggregate to daily means; summarise each $N$ by the mean over draws with a **block-bootstrap 95 % band** over days and draws.
5. **Second axis:** support duration $D\in\{7,14,30,60\}$ days at fixed $N=5$ → a sparsity *surface*.
6. **Own-history mode** (forecast the future at stations in $A$) is run as a secondary panel.

**Outputs.** (a) CRPS vs $N$ per method (log-scaled $N+1$ axis) with bands; (b) CRPSS vs M2; (c) summary parameters from fitting $\mathrm{CRPS}(N)=c_\infty+(c_0-c_\infty)\,e^{-N/\nu}$ per method ($\nu$ = sample-efficiency scale); (d) $N^\*$ = smallest $N$ with CRPSS vs M2 ≥ 0.10; (e) the support-duration surface.
**Caveats made explicit in the figure:** M4 is undefined at $N=0$ (no data) and is shown as untrainable; the plan's expectation of an "exponential decay" for scratch models and a "flattened curve" for transfer are *hypotheses* the figure tests, not shapes to be drawn.
**Compute:** adaptation is seconds-to-minutes per (N, draw); the experiment is sharded by $N$ across matrix jobs and cached.

## 8. Statistical inference

* **Diebold–Mariano.** For two methods, test $H_0:\mathbb E[d_t]=0$ on the daily CRPS differential $d_t=\mathrm{CRPS}^A_t-\mathrm{CRPS}^B_t$ with a HAC (Newey–West) variance, bandwidth $=\lceil h/24\rceil+2$ days, and the Harvey–Leybourne–Newbold small-sample correction (Diebold & Mariano, 1995; Harvey et al., 1997).
* **Intervals.** Stationary block bootstrap (Politis & Romano, 1994), mean block 7 days, 2,000 replicates, 95 % bounds for CRPS, CRPSS and coverage. Resample **days**, not station-days, because stations are strongly cross-correlated within a day.
* **Effective sample size.** A 90-day season gives ≈ 90 days but, with 72-hour forecast autocorrelation, an effective ~25–30 independent days. Report effective $n$ beside every p-value; one season cannot establish regime-independent conclusions.
* **Effect sizes first.** Report CRPSS with intervals before p-values.

## 9. Prospective shadow-run protocol

**Why.** Historical backtesting is vulnerable to leakage and tuning; the smog season (late October through January) provides a genuine out-of-sample test. This is the plan's "shadow run", made operational:

1. **Freeze register.** Before the first scored issuance, record the git tag, `config_hash`, bundle sha256 and the pre-registered `configs/evaluation.yaml`. Anything changed afterwards is a *new model version* with its own start date; scores are never recomputed retroactively.
2. **Baselines from day 1.** The pipeline logs M0–M3 forecasts for every issuance from the first scheduled run, so the season is never lost while the learned model is still being built.
3. **Challenger mode.** A candidate bundle is logged but not published (`method` ≠ the published one) until promoted; promotion is a reviewed PR.
4. **Scoring.** `shadow-scoring.yml` runs daily: *provisional* scores at window end + 6 h from the API; *settled* scores at +96 h from the archive. Both go to the append-only `score_log`. Score the first valid issuance prospectively. Reruns are logged but not scored. Score the first valid issuance prospectively. Reruns are logged but not scored.
5. **Live vs replay shadow.** *Live* = forecasts generated before the outcome existed. *Replay* = a model finished later is run over logged `state/inputs` snapshots of earlier days; its training data must end **before** the replayed period and its config must be frozen before the replay starts. Replay and live results are always reported separately, replay marked as such.
6. **Monitoring vs testing.** Interim looks (coverage, completeness, mode counts) are for operations. Hypothesis tests H1–H6 are run **once**, after the season, on settled scores.
7. **Reporting.** Rolling 30-day scorecard on the public accuracy page with sample sizes; end-of-season report with every pre-registered analysis, including failures and the distribution of degradation modes and of $N$.
8. **Limits.** One season is one climate realisation; Lahore's network changes during the season; results from provisional truth are reported only as provisional.

## 10. Reporting artifacts

| Artefact | Produced by | Where |
|---|---|---|
| Ladder table (M0–M8 × horizon × {CRPS, CRPSS, coverage, IS, twCRPS}) | `eval ladder` | `reports/ladder.md` (git-ignored), figure copies in `docs/figures/` |
| Sensor-sparsity curve + surface + $\nu$, $N^\*$ | `eval sparsity` | figure + CSV |
| Reliability diagrams, PIT-style coverage curves, sharpness | `eval report` | figure |
| Event verification (Brier, POD/FAR/CSI/PSS) | `eval report` | table |
| Season report with H1–H6 verdicts | `eval report` | markdown → thesis |
| Public scorecard | `site build` from `score_log` | `site/accuracy/` |

**Negative-results policy.** If M6 ties M7, or transfer from Delhi is negative at $N=0$, that is reported with the same prominence as a positive result, and the corresponding ADR is updated.
