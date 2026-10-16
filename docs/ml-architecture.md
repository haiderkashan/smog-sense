# ML Architecture

> **Audience:** researchers and engineers implementing Phases 2–3 (revision 3). Math renders on GitHub. Every design choice that departs from the original research plan is marked
> **Δ** and justified in the [verification log](verification-log.md).

## 1. Problem formulation

**Notation.** Stations $s\in\mathcal S_d$ belong to a domain $d$ (target: Lahore; source: Delhi). Issuance times $T$ lie on a UTC lattice (training: 00/06/12/18 h; operations: 00 h). $x_{s,t}$ is quality-controlled hourly PM2.5 (µg m⁻³)
with validity $m_{s,t}\in\{0,1\}$ ($1$ = valid and *not* imputed). Horizons $h\in\mathcal H=\{24,48,72\}$.

**Target (Δ).** The original plan forecasts the concentration "at time $T+h$". Hourly low-cost readings are noisy, AQI categories are defined on 24-hour means, and CAMS is a 40 km model whose hourly skill is limited, so SmogSense forecasts the **24-hour block mean**:

$$
y_{s,T,h}=\frac1{|V|}\sum_{t\in V}x_{s,t},\qquad V=\{\,t\in[T+h-24\,\mathrm h,\;T+h):m_{s,t}=1\,\},\qquad y_{s,T,h}\ \text{defined iff}\ |V|\ge 18 .
$$

(Hour-start labels; the block is the 24 hours $T+h-24,\dots,T+h-1$. The $T+24,T+48,T+72$ of the plan are the block *ends*.)

**City target.** The city value is a *pseudo-location*: its target is the mean of the group-collapsed station block means (one value per co-located group, reference monitor first, otherwise the median of corrected low-cost sensors) over a **panel frozen at the season freeze**, requiring ≥ 3 valid stations. Its quantiles are modelled directly, not composed from station quantiles. The bulletin states the panel size and the reference/low-cost split.

**Probabilistic output.** For each $(s,T,h)$ the product returns quantiles $\hat q_{s,T,h}(\tau)$ for $\tau\in\mathcal Q=\{0.05,0.10,\dots,0.95\}$ ($K=19$). The **public** levels are $0.10,0.50,0.90$: an **80 % central interval** $[\hat q_{.10},\hat q_{.90}]$ (not 90 %) and a
median. $\hat q_{.90}$ is *not* a worst case; about one day in ten exceeds it ([dissemination](dissemination-and-ui.md#2-risk-communication-rules-for-probabilistic-forecasts)).

**Transform.** Models are trained on $z=\log(1+y)$. Quantiles commute with monotone maps, $Q_\tau[g(Y)]=g(Q_\tau[Y])$, so $\hat q=\exp(\hat z)-1$ estimates the same population quantile while stabilising optimisation of the heavy right tail. All scoring is in µg m⁻³.

**Inputs available at $T$** (single as-of rule, [data engineering §11](data-engineering.md#11-temporal-alignment)): observation history from the *available* stations $A$ ($|A|=N$, possibly $0$) with the last $\approx L_{\rm oaq}$ hours missing by design; CAMS stitched lookback and the forecast covariates for the three blocks; the **forecast age** $\delta=T-B^\*(T)$; fire exposure; static station descriptors; calendar.

## 2. Feature engineering (physics-informed)

### 2.1 Why ventilation: a box-model derivation

For a well-mixed column of height $H$ (boundary-layer height, PBLH) and length $L$ in the wind direction, with emission flux $Q$, mean transport wind $\bar u$ and first-order loss rate $\lambda$:

$$
\frac{dC}{dt}=\frac{Q}{H}-\frac{\bar u}{L}\,C-\lambda C
\quad\Longrightarrow\quad
C^{*}=\frac{Q/H}{\bar u/L+\lambda}\;\approx\;\frac{Q\,L}{H\,\bar u}=\frac{QL}{VC}\quad(\lambda L\ll\bar u).
$$

Hence $\ln C^{*}\approx\ln(QL)-\ln VC$: concentration is **inversely proportional to the ventilation coefficient**, and *additive in log space*. This is the physical reason to feed $\ln VC$ and the *mean of $1/VC$* (not the mean of $VC$) to the model.

### 2.2 Feature definitions

| Group | Feature (per forecast block unless stated) | Definition |
|---|---|---|
| **Ventilation** | $VC(t)$ | $VC=\max(\mathrm{PBLH},50\,\mathrm m)\cdot WS$ with $WS=\sqrt{u_{10}^2+v_{10}^2}$, and a floor $VC\ge10\ \mathrm{m^2s^{-1}}$. Optional $VC_{925}$ with 925 hPa wind (a better proxy for the mixed-layer mean wind; verify availability in the Phase 1 contract test) |
| | $\ln VC$ min / mean / night-mean | Night = 18:00–06:00 Asia/Karachi |
| | **Dilution index** | $DI=\frac1{24}\sum_{t\in\text{block}}\frac{1}{VC(t)}$ — the box-model quantity |
| | **Stagnation run-length** | longest run of consecutive hours with $VC<\theta$, $\theta\in\{1000,2000,4000\}\ \mathrm{m^2s^{-1}}$ (candidate cut-offs, ablated — not asserted) |
| **Stability** | $\Delta\theta$ | $\theta_{925}-\theta_{2m}$ with $\theta=T\,(1000/p)^{0.286}$; positive ⇒ stable/inversion |
| **Humidity** | RH, fog flag | Magnus: $e_s(T)=6.1094\exp\!\big(\tfrac{17.625\,T}{T+243.04}\big)$ hPa, $RH=100\,e_s(T_d)/e_s(T)$; fog if RH > 90 % |
| **Removal** | block precipitation sum | wet scavenging proxy |
| **Radiation** | block mean surface solar radiation | boundary-layer growth proxy |
| **CAMS as covariate** | CAMS PM2.5 block mean; CAMS AOD550 | the physics-based baseline *is* an input |
| **Forecast age** | `cams_lead_offset_h` ($\delta = T - B^\*$) | 12 / 18 / 12 / 18 h for $T$ = 00 / 06 / 12 / 18 Z at level 0, **+12 h at degradation level 1**. Training adds a stale-cycle offset $\{0,12,24\}$ h with probabilities $\{0.70,0.20,0.10\}$. An input to M3b and the M5 base (M6/M7 inherit it through the base) |
| **Recent CAMS bias** | trailing 7-day mean of $(x-\mathrm{CAMS})$ at the station | input to the Stage-2 residual head (M6/M7) and to M3b; **excluded from the M5 base**, which must stay free of local calibration (representativeness: 40 km grid vs a point) |
| **Fire** | ring counts/FRP, $E_{24},E_{48},E_{72}$ | §3 |
| **Calendar** | sin/cos of local hour, weekday, day-of-year; event flags | Event flags (Diwali) exist for Indian domains only and are **excluded from every transferred model** (M5, M6, M7); they appear only in Delhi-only diagnostics |
| **Static** | elevation, distance to centre (optional built-up fraction, road density); **absolute lat/lon only in within-domain models (M4)** | so unseen stations are not anonymous without giving a Delhi-trained tree a coordinate that places Lahore in an edge leaf |

### 2.3 Lagged cross-correlation (diagnostic, not a leakage vector)

To check that the fire kernel's time scales are physically plausible, compute on *deseasonalised* series from the **training seasons only**

$$
\rho(\ell)=\frac{\sum_t\big(E_{t-\ell}-\bar E\big)\big(\tilde x_t-\bar x\big)}{\sqrt{\sum_t(E_{t-\ell}-\bar E)^2}\sqrt{\sum_t(\tilde x_t-\bar x)^2}},\qquad \ell=0,\dots,96\ \mathrm h,
$$

with stationary-bootstrap confidence bands. The maximising lag $\ell^\*$ should agree with the typical advective time $d/(3.6\,\bar w)$ ($\approx 8$–$40$ h for $100$–$300$ km at $2$–$4$ m s⁻¹). The same analysis is run for upstream CAMS PM2.5. Selected lags are frozen before touching validation or test data.

### 2.4 Tensors, masks and scaling

Two consumers, two input sets. The **base** (§6) consumes the tabular features of §2.2 (ventilation, stability, CAMS PM2.5 and AOD550, fire exposure, forecast age, calendar, static descriptors). The **sequence encoder** (§4) consumes only *local observation* channels: $X\in\mathbb R^{L\times C}$ with $L=72$ and $C=6$ — PM2.5 $x\odot m$, the hourly *observed − CAMS* error (CAMS from the as-of stitched lookback), RH and temperature (the sensor's own where present, otherwise CAMS 2 m values), the validity mask $m$, and the time since the last observation $\delta_t$ (GRU-D-style inputs after Che et al., 2018; $\delta_t$ is *not* the forecast age). CAMS physics and fire reach the model through the base only, which is the point of the split (Δ: revision 2 fed ten channels to the encoder; `configs/features.yaml` must list these six).
Scalers (robust, median/IQR) are fit on the **source-domain E-block training data only** (§7) and then frozen: re-fitting them on Lahore would silently erase the very domain shift that adaptation must learn, and would change the meaning of inner-loop gradients.

## 3. Fire transport exposure

**Δ — target-centric, wind-aligned; not a fixed upwind box.** The plan places a box "upwind of Lahore" using north-westerly winds from Indian Punjab. That geometry suits Delhi (downwind of the burn belt) but not Lahore: Indian Punjab lies **east and south-east** of Lahore (Amritsar ≈ 50 km east), so north-westerly flow brings Pakistani-Punjab air, and Indian-Punjab smoke arrives under easterly/south-easterly flow.
Rather than hard-coding a direction, define exposure relative to each target so the same definition transfers between cities and the model discovers which sectors matter.

**Single angle convention.**
* $\theta_i$ = azimuth from the **target to** detection $i$ (clockwise from north). Used only to bin detections into the eight 45° sectors.
* $\beta_i=\theta_i+180^\circ \pmod{360^\circ}$ = bearing of the vector **fire → target**.
* $\phi(t)=\operatorname{atan2}(u,v)$ = the direction the air moves **toward**, clockwise from north ($u$ east, $v$ north components).
* $a_i(t)=\max\{0,\cos(\phi(t)-\beta_i)\}$: 1 when the air is moving exactly from the fire towards the target, 0 when it moves away or sideways.

**Worked example (Amritsar → Lahore).** Lahore 31.5204°N, 74.3587°E; a fire at Amritsar 31.634°N, 74.8723°E. Offset ΔE ≈ 48.8 km, ΔN ≈ 12.6 km, so $\theta\approx75.5^\circ$ and $\beta\approx255.5^\circ$.
Easterly wind (wind *from* the east, $u=-5$ m s⁻¹): $\phi=270^\circ$, $a=\cos(14.5^\circ)\approx0.97$ — high exposure. Westerly wind ($u=+5$): $\phi=90^\circ$, $\cos(-165.5^\circ)\approx-0.97$, clipped to $a=0$.

For detection $i$ with FRP $f_i$ (MW), platform weight $\omega_i=1/n_{\text{plat}}(\tau_i)$, acquisition time $\tau_i$, distance $d_i$ (km), and target wind speed $w(t)=\max(\lVert\mathbf u(t)\rVert,w_{\min})$:

$$
\Theta_i(t)=\frac{d_i}{3.6\,w(t)}\ \ [\mathrm h],\qquad
K_i(t)=a_i(t)\,e^{-\Theta_i(t)/\tau_{\rm decay}}\;e^{-(\min(t,T)-\tau_i)^{+}/\tau_{\rm age}},
$$

$$
E(t)=\sum_{i:\ \tau_i\le T}\omega_i\,f_i\,K_i(t),\qquad
E_h=\frac1{24}\sum_{t\in[T+h-24,\,T+h)}E(t).
$$

For $t>T$ the wind comes from the **CAMS forecast** and the fire set is frozen at the detections knowable at $T$ (`acq_datetime + 3 h ≤ T`; persistence of the recent burning rate). Defaults: $\tau_{\rm decay}=24$ h, $\tau_{\rm age}=48$ h, $w_{\min}=0.5$ m s⁻¹, chosen on source seasons and then frozen.
The compact feature set is the four ring counts, four ring FRP sums and $E_{24},E_{48},E_{72}$; the full set adds ring × sector counts. **Interpretability output:** mean |SHAP| by sector, per domain — the expected Lahore signature is easterly/south-easterly weight in October–November; its absence would be a red flag worth investigating.
**Sensor continuity:** all counts and FRP are platform-normalised so that the Suomi-NPP retirement (1 Nov 2026) and the MODIS decline do not create artificial trends.

## 4. Sequence encoder

**GRU** (Cho et al., 2014), two layers, hidden 64, over the six local channels of §2.4:

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z),\quad r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r),\quad
\tilde h_t=\tanh\!\big(W_hx_t+U_h(r_t\odot h_{t-1})+b_h\big),\quad
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t ,
$$

followed by a linear map of $h_L$ to a latent $\mathbf z\in\mathbb R^{32}$ that feeds the residual head (§5).
**Parameter count (exact, asserted by a unit test).** GRU layer 1: $3\,(64\cdot6+64\cdot64+2\cdot64)=13{,}824$; layer 2: $3\,(2\cdot64\cdot64+2\cdot64)=24{,}960$; projection $64\to32$: $2{,}080$; encoder total **40,864**. Residual head $46\to64\to57$: $3{,}008+3{,}705=6{,}713$. The whole neural stack is therefore **47,577 parameters (≈ 48 k)**, plus 14 learned step-size scalars (§8). The count changes if the channel set or the head input changes, and the test constant must then be updated together with the documentation. (Revision 2 quoted 42,016 for a 12-channel encoder and "≈ 65 k" for the whole stack; both are superseded.)
**TCN** (Bai et al., 2018) is an optional alternative configuration (cut from the planned scope, see the roadmap): causal dilated convolutions $(F\!*\!x)(t)=\sum_{k=0}^{K-1}f(k)\,x_{t-dk}$ in residual blocks. With kernel $K=3$, $n_c=2$ convolutions per block and dilations $\{1,2,4,8,16\}$ the receptive field is

$$
R=1+n_c\,(K-1)\sum_i d_i=1+2\cdot2\cdot31=125\ \mathrm h\ \ge\ L=72 .
$$

The encoder is a configuration switch; the ladder (§10) decides whether a sequence encoder earns its place over the physics base alone (M5).

## 5. Quantile loss, the residual head, and the quantile function

**Pinball (quantile) loss** (Koenker & Bassett, 1978): $\rho_\tau(u)=u\,(\tau-\mathbf 1\{u<0\})=\max\{\tau u,(\tau-1)u\}$. For $Y\sim F$:
$\frac{d}{dq}\mathbb E\,\rho_\tau(Y-q)=F(q)-\tau$, so the minimiser is $F^{-1}(\tau)$. The penalty ratio for under- vs over-prediction is $\tau/(1-\tau)$ — $9{:}1$ at $\tau=0.9$, as the plan notes.

**Δ — Physics-Residual Meta-Quantile Learning.** The frozen physics base (M5, §6) forecasts the regional distribution; the adapted neural network predicts the *local residual on top of it*, for **every** level and horizon:

$$
\hat z^{(7)}_{\tau,h}\;=\;\hat z^{(5)}_{\tau,h}\;+\;\Delta\hat z_{\tau,h}\Big(\mathrm{GRU}_{\theta'}(X_{72\,\mathrm h}),\ \mathbf u\Big),\qquad \tau\in\mathcal Q\ (K=19),\ h\in\mathcal H=\{24,48,72\},
$$

with $\hat z^{(5)}$ the base quantile in log1p space. M6 has the same form with its own weights (supervised initialisation); the headline candidate is M7. The **head input** is $[\mathbf z\ (32)\,\|\,\text{recent CAMS bias}\ (1)\,\|\,\text{station RH at }T\ (1)\,\|\,\ln VC\text{ block means}\ (3)\,\|\,\hat z^{(5)}_{\tau,h},\ \tau\in\{0.10,0.50,0.90\},\ h\in\mathcal H\ (9)]$: 46 numbers. The head is an MLP $46\to64$ (ReLU) $\to57$ (linear); the **last layer is zero-initialised** so that $\Delta\hat z\equiv0$ and an untrained calibrator reproduces M5 exactly. The head sees only three base quantiles per horizon, enough to say where the base puts the day and how wide it is, without a 57-dimensional input. Loss over a batch $B$ with validity masks $\mu_{s,T,h}$ (the base prediction is a constant input; no gradient reaches the trees):

$$
\mathcal L(\theta)=\frac1{\sum\mu}\sum_{(s,T,h)\in B}\mu_{s,T,h}\;\frac1{19}\sum_{\tau\in\mathcal Q}\rho_{\tau}\Big(z_{s,T,h}-\hat z^{(5)}_{\tau,h}-\Delta\hat z_{\tau,h}\Big).
$$

The calibrator is therefore **part of the published forecast** (unlike revision 2's temporary head), and the product quantiles are $\hat z^{(7)}$ after monotone rearrangement (below). If adaptation is rejected or skipped, $\Delta\hat z\equiv0$ and the published forecast is M5.
**Tails.** Every one of the 19 levels, including 0.05 and 0.95, is now trained directly, so revision 2's "untrained-tail" limitation no longer applies in that form. What remains is *sparse tail supervision*: a 14-day support window contains about one exceedance of $q_{.95}$ (risk R15). The optional extension of the grid to $\tau=0.025/0.975$ stays an owner decision, not applied.

**Quantile function object.** Interior: piecewise-linear in $\tau$ through the 19 knots. Tails: upper exponential, $q(\tau)=q_K+s_{hi}[\Lambda(\tau)-\Lambda(\tau_K)]$ with $\Lambda(\tau)=-\ln(1-\tau)$ and
$s_{hi}=\frac{q_K-q_{K-1}}{\Lambda(\tau_K)-\Lambda(\tau_{K-1})}$; lower, log-linear $q(\tau)=\max\{0,\,q_1+s_{lo}[\ln\tau-\ln\tau_1]\}$. This object provides `ppf`, `cdf`, $P(Y>\theta)$, $P(\text{category})$ and exact-integral CRPS.
**Measured:** against analytic CRPS of lognormal ($\sigma\in\{.3,.5,.8\}$) and gamma distributions, the estimator's relative error is **0.2 % mean (≤ 0.5 % max) at $K=19$ but 1.0–3.5 % mean (up to 6.2 % max) at $K=7$**, and 13 % for a naïve 7-level estimate without a tail model. That is why the scoring grid has 19 levels.

**Monotone rearrangement.** Independently trained quantiles (the 57 boosters of §6, and the 57 residual-adjusted outputs of the calibrator) can cross (in a synthetic test, 7 of 500 rows). Sorting the $K$ values restores monotonicity and **never increases the total pinball loss**: $f(\tau,q)=\rho_\tau(y-q)$ has $\partial^2f/\partial\tau\,\partial q=-1<0$ (submodular), so by the rearrangement inequality $\sum_kf(\tau_k,q_{\pi(k)})$ is minimised by the sorted pairing (cf. Chernozhukov et al., 2010). Rearrangement is applied to $\hat z^{(7)}$ at inference and inside the acceptance check; training uses the unsorted values.

## 6. LightGBM quantile ensemble (the frozen physics base)

**Δ — why the order is flipped.** Revision 2 adapted a neural encoder and fed its frozen latent to a tree stacker. That made the trees depend on the *shape* of the adapted latent, which meant simulated deployments, a drift guard and an encoder↔stacker binding. Revision 3 puts the trees first and the adapted network second: gradient-boosted trees learn the regional physics once, on Delhi, and are then **frozen**; the neural network, which MAML can adapt because it is differentiable, learns only the *local residual*. Trees do what they are good at (abrupt, non-linear interactions such as "high upwind fire matters only when the wind aligns **and** the boundary layer is shallow"); the network does what it is good at (reading a short local history). The fallback is exact: $\Delta\hat z=0$ gives M5.

**No native multi-quantile in LightGBM.** The plan states that LightGBM "natively supports multi-quantile regression" through an `alpha` array. It does not: `alpha` is a single `double`; passing `[0.1, 0.5, 0.9]` raises *"Parameter alpha should be of type double"* (tested, LightGBM 4.7). SmogSense trains $3\times19=57$ boosters with `objective=quantile`, `alpha=<τ>`, then applies rearrangement (§5).

**Inputs (M5).** All tabular features of §2.2 — ventilation (ln VC min/mean/night-mean, dilution index, stagnation run-lengths), stability $\Delta\theta$, RH and fog flag, precipitation, radiation, CAMS PM2.5 and AOD550, forecast age $\delta$, fire (rings, $E_{24},E_{48},E_{72}$), local-time calendar, elevation and distance to centre — **except** (i) the trailing 7-day *recent CAMS bias*, so that M5 is pure physical transfer with no local calibration, and (ii) every **domain-identifying** feature: absolute latitude/longitude and Indian-only event flags (Diwali). A tree trained on Delhi coordinates places Lahore in an edge leaf and returns an arbitrary shift, and an event flag that does not exist in Lahore would have to be zero-filled or dropped at inference, silently changing the model. Within-domain models (M4) may use coordinates. Fire and AOD feature groups are each masked with probability 0.1 during training (LightGBM missing values), so a failed FIRMS call degrades gracefully.

**Regularisation for tails.** A leaf should hold at least two expected tail observations: $n_{\rm leaf}\cdot\min(\tau,1-\tau)\ge2\Rightarrow$ `min_data_in_leaf` $\ge40$ at $\tau=0.05$ (the default).

**Out-of-fold base predictions (C1, C2).** The calibrator must be trained on base predictions that look like deployment, i.e. *out-of-sample for the trees*. Trees fit on block E would predict E almost perfectly (in-sample), so the calibrator would learn that the base is nearly always right and would under-correct on Lahore. Therefore: split the seasons of block E into $K=\min(3,\#\text{seasons in E})$ **season-blocked folds** with a 72-hour embargo and purging at every boundary; for each fold fit the 57 boosters on the other folds and predict the held-out fold; the concatenated predictions are the **out-of-fold (OOF) base** used as the calibrator's input and baseline during pre-training and meta-training. The **final base** (the product M5) is fit on all of E with early stopping on block V. Unlike encoder latents, tree outputs live in one shared, physically meaningful space (log1p µg m⁻³), so cross-fitting is valid here; the revision-2 rejection of K-fold cross-fitting (V-45) applied to incoherent latent frames and still stands for them. At deployment the calibrator sees the final base, which is slightly better than the OOF bases it was trained on: a conservative mismatch, reported rather than corrected.

**Binding (C3).** The model bundle stores `base_sha256` (the final base's hash) and the partition hash of the OOF folds in the calibrator manifest; loading refuses a mismatched pair (degradation level 2). **M6 and M7 each own a calibrator; both sit on the same M5 base**, which is a tabular bundle with no neural part.
**No neural/GBM blend and no warm-start boosting in the product path.** Both are kept as ablations (A-warm, A-blend) because the plan proposed them.

**Fallback rule.** If the calibrator is not robust (it fails the residual-bound or acceptance checks in simulation on block V or on the Lahore dev window) or adaptation adds nothing over $\Delta\hat z=0$, **ship M5 as the headline and report the negative result** with the same prominence as a positive one.

## 7. Sparse-network task construction

**Δ — the plan conflates two different "N"s.** In the plan's sensor-sparsity experiment, "zero local stations" is described as pure zero-shot transfer, but the calibrator consumes the *target station's PM2.5 history*; with no local station there is no history to consume. The consistent definition: $N$ is the number of **available stations**, which supply *both* the labels used for adaptation *and* the observed-history input at inference.

**Source-domain season blocks (E/V/T).** The source seasons (Delhi) are split chronologically into three non-overlapping blocks, with a 72-h embargo and purging at every boundary. Revision 2's block S (stacker rows from simulated deployments) is gone; E now also supplies the cross-fit folds of §6.

| Block | Role | Example, 8 seasons |
|---|---|---|
| **E** | scalers; Stage-1 base (cross-fitted into $K=\min(3,\#\text{seasons})$ season folds); supervised pre-training; meta-training | oldest 6 |
| **V** | tuning, early stopping, final-base early stopping, residual-bound threshold $\rho_{99}$ | 1 |
| **T** | Delhi confirmatory test (H5), untouched until the freeze | newest 1 |

The Phase 1 audit rule "≥ 5 seasons ⇒ Delhi-only source is viable" is unchanged (E then holds ≥ 3 seasons, enough for $K=3$ folds; with exactly 4 seasons E holds 2 and $K=2$).

**Task.** A **task** $\mathcal T=(d,p,A,B,W^{\rm sup},W^{\rm qry})$: domain $d$; season $p$ within block E (V for threshold simulation); available set $A\subset\mathcal S_d$ with $|A|=N\sim\mathrm{Unif}\{0,1,2,3,5,8\}$; held-out set $B=\mathcal S_d\setminus A$ (capped for speed); an earlier support window and a later query window separated by a 72-hour embargo. Base predictions inside a task are the **OOF** predictions of §6.
**Twin-aware:** stations in the same `colocated_group` always fall on the same side of the $A/B$ split (a held-out twin of an available station would leak its readings). Station identity is `sensor_id`.

* **History input.** For any target $s$: $H_A(s,t)=\sum_{a\in A}w_a(s)\,x_{a,t}\big/\sum_aw_a(s)$ with inverse-distance weights $w_a(s)\propto\mathrm{dist}(s,a)^{-2}$ within 25 km (plus a "distance to nearest available" feature). The six-channel tensor of §2.4 is built from $H_A$ (or from the station's own series in own-history mode).
* **Two evaluation modes.** *Own-history* (forecast the future at stations in $A$) and *neighbour-history* (forecast at $B$ using $A$ — i.e. at places without a sensor). Training samples draw "own history present" with probability 0.5. The sparsity curve's primary mode is **leave-stations-out** (neighbour-history).
* **Group dropout.** Fire and AOD feature groups are each masked with probability 0.1 *when training the base* (§6). The whole PM2.5 history is **no longer** dropped as a modality: sensor-free mode is the base by construction (C7), not a trained behaviour of the network.
* **Baselines use the available set only.** M0, M2, M3 and M3b see only $A$ (support window and history). **M5 uses no station data at all** (no recent-bias feature), so it is independent of $A$: at every $N$ it returns the same forecast, and it is the forecast of every learned method at $N=0$. At $N=0$ only M1 (deterministic) and M5 (with M6/M7, which equal it) are defined.
* **Explicit $N=0$ behaviour (C7).** With no support there is nothing to adapt on: $\Delta\hat z=0$, the published forecast is M5, the degradation level is 0 and `adaptation_status = skipped_n0`. If $N\ge1$ but the support window has too few valid days (default: fewer than 5, `adaptation.min_support_days`, to be fixed by the threshold pilot), the same fallback applies with `skipped_insufficient_support`.

Meta-training tasks come from every Delhi station-season in block E (≈ hundreds). Optional extra source domains (Amritsar, Ludhiana) add task diversity if Delhi-only transfer is brittle.

## 8. Meta-learning (MAML, Meta-SGD) and adaptation

**Setting.** Neural parameters $\theta=(\theta_{\rm gru},\theta_{\rm head})$: the six-channel GRU of §4 and the zero-initialised 57-output head of §5, adapted together by default (`adapt_scope: all`). Task loss $\mathcal L_{\mathcal T}(\theta;D)=\frac1{|D|}\sum_{(x,y)\in D}\ell\big(f_\theta(x),y\big)$ with $\ell$ the masked mean pinball loss over all 19 levels and 3 horizons of $\hat z^{(7)}=\hat z^{(5)}+\Delta\hat z$ (§5).

**Loss and clipping (C8).** The inner and outer loss is **pinball**; gradient-norm clipping at **1.0** in both loops. Huber / quantile-Huber is an **ablation**, not the default: pinball is the proper score for the target, and heavy-tail robustness is better handled by the clip.

**Meta-SGD inner loop (Δ).** The fixed scalar rate $\alpha=0.01$ of revision 2 is replaced by a **meta-learned step-size vector** $\boldsymbol\alpha$ (Li et al., 2017), updated by the outer Adam loop together with $\theta$. Default granularity: **one step size per parameter tensor** (14 scalars: 8 GRU tensors, 2 projection, 4 head), parametrised $\alpha_k=\operatorname{softplus}(\rho_k)\ge0$ and initialised at 0.01; **per-parameter** step sizes (47,577 extra meta-parameters) are an ablation, because with a few hundred source tasks they can overfit and a negative $\alpha$ (allowed in the original Meta-SGD) can make the inner loop diverge. With $J$ steps (support set $D_i^{\rm sup}$):

$$
\theta_i^{(0)}=\theta,\qquad \theta_i^{(j+1)}=\theta_i^{(j)}-\boldsymbol\alpha\odot\nabla_{\theta_i^{(j)}}\mathcal L_{\mathcal T_i}\big(\theta_i^{(j)};D_i^{\rm sup}\big),\qquad \theta_i'=\theta_i^{(J)} .
$$

**Outer loop** (query set $D_i^{\rm qry}$, later in time than the support set; $\phi=(\theta,\boldsymbol\rho)$):

$$
\min_{\phi}\ \sum_{\mathcal T_i\sim p(\mathcal T)}\mathcal L_{\mathcal T_i}\big(\theta_i';D_i^{\rm qry}\big),\qquad
\phi\leftarrow\phi-\beta\,\nabla_\phi\sum_i\mathcal L_{\mathcal T_i}\big(\theta_i';D_i^{\rm qry}\big)\ \ (\text{Adam}).
$$

**Gradient structure** (Finn et al., 2017). With $H_i^{(j)}=\nabla^2_{\theta}\mathcal L_{\mathcal T_i}(\theta_i^{(j)};D_i^{\rm sup})$,

$$
\nabla_\theta\,\mathcal L^{\rm qry}_i(\theta_i')=\Big[\prod_{j=0}^{J-1}\big(I-\operatorname{diag}(\boldsymbol\alpha)H_i^{(j)}\big)\Big]\nabla_{\theta_i'}\mathcal L^{\rm qry}_i(\theta_i') .
$$

The plan's "gradient of the gradient" is exactly the Hessian product. **Algorithms** (config `meta.algorithm`): **FOMAML** (default) sets $H\approx0$; the step-size gradient then keeps its first-order form $\partial\mathcal L^{\rm qry}/\partial\alpha_k\approx-\sum_{j}\big\langle\nabla_{\theta_k}\mathcal L^{\rm sup}(\theta^{(j)}),\ \nabla_{\theta_k}\mathcal L^{\rm qry}(\theta')\big\rangle$, which ignores how later support gradients depend on earlier steps. *Reptile* (Nichol et al., 2018) and *second-order MAML* ($J\le3$) remain implementable but are **cut from the planned scope**.
**Default: first-order, $J=3$, learned $\boldsymbol\alpha$ frozen at deployment**, on CPU via `torch.func.functional_call`.
**Adaptation scope is an ablation (C4).** ANIL (Raghu et al., 2020) adapts only the head. In revision 2 that left the latent unchanged and served as a negative control; here the head *produces the correction directly*, so head-only adaptation can change the forecast and is a legitimate cheaper competitor. The ablation compares `all` (default), `head_only` (ANIL) and `gru_only`; if a smaller scope matches `all`, the simpler scope is preferred.

**Synthetic low-cost sensor corruption (task augmentation, Δ).** Delhi's regulatory stations carry none of the humidity and gain artefacts of Lahore's low-cost network. During meta-training (and M6 pre-training), each task $i$ draws one set of artefacts and corrupts the PM2.5 channel of its available stations:

$$
\tilde x_{\rm PM2.5}(t)=a_i\,x_{\rm PM2.5}(t)+b_i\,\max\{0,\ \mathrm{RH}(t)-70\}+c_i,\qquad a_i\sim\mathcal U(0.8,1.3),\ b_i\sim\mathcal U(0,1.5),\ c_i\sim\mathcal U(-10,25).
$$

The draw is fixed across the 72-h window and across the task's support and query windows (a device characteristic), the observed − CAMS error channel is recomputed from $\tilde x$, RH itself is not corrupted, and the **query labels $y$ stay clean** (regulatory truth). The intent is that the inner loop learns to strip task-specific humidity and gain bias in $J=3$ steps, which pooled pre-training (M6) cannot do. Four conditions keep this honest:
1. **M6 is trained with the identical augmentation** (a fresh draw per training sequence), so the M7-vs-M6 difference isolates the *inner loop*, not the augmentation.
2. **Label policy is explicit.** `corruption.support_labels` is `clean` (default, as specified) or `same_as_inputs`. With clean support labels the artefact is identifiable from support; at a Lahore low-cost station, however, the support labels come from the *same sensor*, so a deployment-faithful variant corrupts them too, and then the artefact is *not* identifiable from support alone. Which variant transfers is an empirical question (experiment X-11), not an assumption.
3. **The ranges are priors.** They are to be checked against the inverse of the per-sensor corrections fitted in the Phase 1–2 co-location audit ([data engineering §5](data-engineering.md#5-quality-control-and-low-cost-sensor-handling)); a gain of 0.8–1.3 under-covers the roughly 2× over-reading reported for some optical sensors. `corruption.p` (default 1.0) allows a fraction of clean tasks, which matters because Lahore's reference monitors have no artefact (risk R17).
4. **Evaluation rows are never corrupted**, except in the separate synthetic condition H5-C ([evaluation §1](evaluation-strategy.md#1-evaluation-questions-and-hypotheses)); stored data are never modified.

**One shared recipe (C5).** A single config block `adaptation.inner` (loss, learned $\boldsymbol\alpha$, $J$, support window, masks, clip, `adapt_scope`) is used by meta-training, threshold simulation on block V, and deployment. M6 uses the same block with a fixed scalar $\alpha=0.01$ in place of the learned vector. The support window is sampled from {7, 14, 30} days in meta-training and deployment uses the same set. **Deployment $J$ equals meta-training $J$**; any other $J$ is an ablation. Adaptation always starts from the *meta-initialisation* (never from yesterday's adapted weights), so it is a deterministic function of the data and cannot drift.

**Deployment adaptation (Lahore): residual bound and acceptance check (C6).**
1. Take the available stations' support window; split **chronologically**: *fit* part (earlier), a **72-h embargo**, and *check* part (latest). Base forecasts $\hat z^{(5)}$ for all of them come from the final base, as-of.
2. Adapt $\theta\to\theta'$ on the fit part with the shared recipe.
3. On the check part compare the pinball loss of the adapted forecast $\hat z^{(7)}$ with that of $\Delta\hat z=0$ (M5). **Accept iff** the improvement is **≥ 2 %** *and* the adapted forecast is better on **≥ 60 %** of check days, **and** the residual statistic $R=\operatorname{mean}|\Delta\hat z|$ (over check-window rows, horizons and levels, log1p units) satisfies $R\le\rho_{99}$.
4. $\rho_{99}$ is the **99th percentile of $R$ over simulated deployments on block V** (draw $N$, $A$ and windows from V tasks, run the same recipe, compute $R$ on the simulated check windows). It is calibrated on source data, never on Lahore. A looser alternative reading — the 99th percentile of the *base's own absolute residuals* on V — would almost never bind, which is why the per-deployment form is used.
5. If accepted, re-adapt on the full support window with the same recipe and publish $\hat z^{(7)}$ from the result (the decision accepts the *recipe* on this network, not one weight vector). If rejected, **$\Delta\hat z=0$ and the published forecast is M5**.
6. Diagnostics, logged and never used for the decision: check-window pinball of M5, of the *unadapted meta-initialisation* ($J=0$) and of the adapted model, and $R$ for both. They separate "meta-initialisation helps" from "adaptation helps" in the RQ3 analysis.

**`adaptation_status`** (logged in `forecast_log` and the run manifest) takes exactly one of: `adapted`, `skipped_n0`, `skipped_insufficient_support`, `rejected_no_gain`, `rejected_residual_bound`. When both rejection tests fail, `rejected_no_gain` is recorded. A rejection is a normal outcome, not a failure, and never changes the degradation level. (`rejected_residual_bound` replaces revision 2's `rejected_latent_drift`: there is no latent frame any more.)

**Expected effect on H5.** Honest expectation: modest on clean Delhi. The corruption augmentation is aimed at artefacts that regulatory data do not contain, so on Delhi block T the M7-vs-M6 gap comes mainly from the meta-learned initialisation and step sizes, and the literature gives real reasons to expect plain fine-tuning (M6) to be competitive. The decisive evidence for the augmentation is H5-C (synthetic corruption on held-out draws) and the Lahore results. A null result on H5 is a finding.

**Memory hygiene.** The adaptation stage must fit the 7 GB runner with margin: it copies *parameters* (not the model) per task, runs under `torch.no_grad` wherever gradients are not needed, caps the support set deterministically (≤ 30 days × available stations, subsampled by a fixed seed if larger), releases tensors and calls `gc.collect()` between tasks, and records its **peak RSS** in the run manifest. A test asserts a peak-memory bound on a synthetic worst-case support set.

**Optional RH mask on support readings (C9; off by default).** When enabled (`adaptation.rh_mask.enabled: true`): drop *support* readings from low-cost sensors whose co-measured RH (device RH if present, otherwise CAMS RH) exceeds **85 %**, provided at least **50 %** of the support window survives; otherwise skip the mask. The mask applies to the *labels used for adaptation only* — never to targets, check windows or evaluation rows. Rationale: optical readings inflate under hygroscopic growth. Risk (R14): high RH coincides with fog and the highest PM, so masking can truncate exactly the upper tail. It is therefore **off** until the mask audit and ablation (RH-stratified scoring: none / 75–85 / above 85) show a benefit.

**Conditions of the residual design (C1–C9).**

| ID | Condition | Where |
|---|---|---|
| C1 | The calibrator trains only on **out-of-fold** base predictions (never on predictions of trees that saw the same rows) | §6 |
| C2 | Cross-fit folds are **season-blocked** with a 72-h embargo and purged lookbacks | §6, §7 |
| C3 | Calibrator is **bound** to its base by `base_sha256`; mismatch ⇒ level 2 | §6 |
| C4 | Adaptation scope (`all` / head-only / GRU-only) is an **ablation**; ANIL is no longer a negative control | §8 |
| C5 | **One shared recipe** block (learned $\boldsymbol\alpha$ frozen at deployment); meta-training $J$ = deployment $J$ | §8 |
| C6 | **Fit / embargo / check** split plus the **residual bound**; rejection ⇒ $\Delta\hat z=0$ | §8 |
| C7 | **Explicit $N=0$** and insufficient-support behaviour: the forecast is M5 | §7 |
| C8 | **Pinball** on all 19 levels with gradient clip 1.0; zero-initialised head; Huber is an ablation | §5, §8 |
| C9 | **RH mask off by default**; audited before any use | §8 |

## 9. Calibration

**CQR** (Romano et al., 2019). With scores $E_i=\max\{\hat q_{lo}(x_i)-y_i,\ y_i-\hat q_{hi}(x_i)\}$ on a calibration set of size $n$ and $\hat Q$ the $\lceil(1-\alpha)(n+1)\rceil$-th smallest, the interval $[\hat q_{lo}-\hat Q,\ \hat q_{hi}+\hat Q]$ covers with probability $\ge1-\alpha$ **under exchangeability**.
Daily time series are not exchangeable, so the target miscoverage is tracked with **adaptive conformal inference** (Gibbs & Candès, 2021):

$$
\alpha_{t+1}=\alpha_t+\gamma\,(\alpha-\mathrm{err}_t),\quad \mathrm{err}_t=\mathbf 1\{y_t\notin C_t(\alpha_t)\},\qquad
\Big|\tfrac1T\sum_{t\le T}\mathrm{err}_t-\alpha\Big|\le\frac{\max(\alpha_1,1-\alpha_1)+\gamma}{\gamma\,T}\ \ \text{for any sequence}.
$$

A horizon-$h$ label arrives $h$ hours late, so each horizon keeps its own $\alpha$-state and is updated when its label becomes available (delayed-feedback variant; the stationary-error guarantee is heuristic under delay and cross-station dependence — hence it is evaluated, not assumed).
**Defaults:** nominal 80 % interval ($\alpha=0.2$); **ACI $\gamma=0.02$**, tuned over $\{0.01,0.02,0.05\}$ on the dev window. With $\alpha_1=0.2$ the bound at $T=90$ days is **1.79 / 0.46 / 0.19** for $\gamma=0.005/0.02/0.05$ — a guarantee only for $\gamma$ large enough. Calibration window 60 days, ≥ 200 samples; lower and upper tails adapted separately. Isotonic recalibration of $\tau\mapsto$ empirical frequency (Kuleshov et al., 2018) is the alternative.
**Status:** calibration is model **M8**, an *optional/backlog* layer: it is accepted only if it improves coverage without worsening CRPS, and it is among the first items cut when time is short. With $n_{\rm eff}\approx28$ the standard error of an empirical 80 % coverage is ≈ 0.076, so coverage claims in the first season are descriptive (H6).

## 10. Model ladder and ablations

Complexity must *earn* its place (the plan itself says the AI is unjustified if it cannot beat CAMS). Each rung is scored with identical data, splits and metrics ([evaluation](evaluation-strategy.md)). "A" is the available station set.

| ID | Method | Purpose / what it isolates |
|---|---|---|
| M0 | Persistence (last complete block *available at T*), residual quantiles from **A** only; undefined at N=0 | naïve floor |
| M1 | Raw CAMS; deterministic at N=0, residual-quantile form from **A** at N≥1 | the physics-based forecast |
| M2 | CAMS-BC fitted on **A** only; equals M1 at N=0 | the *fair* baseline (**reference for H1, H2, H4**): any local data at all |
| M3 | Climatology from **A** only; undefined at N=0 | probabilistic reference without skill |
| **M3b** | **Linear quantile regression** on {CAMS PM2.5, ln VC, RH, recent bias, fire $E_{24}$, δ}, fit on **A**'s support window | simplest learned baseline: does anything beyond a linear post-processor help? |
| M4 | Tabular LightGBM trained on Lahore **A** only | "from scratch" (plan: fails at small N) |
| M5 | **Frozen physics base**: tabular LightGBM trained on Delhi (block E, final fit), physical features only — no recent CAMS bias, no absolute coordinates — independent of **A** | pure physical transfer; a *flat line* in the sparsity curve; the forecast of every learned method at N=0; the **fallback headline** (§6) |
| M6 | M5 + GRU + 57-output residual head, **supervised-pretrained** on OOF-base residuals with the same corruption augmentation, then adapted with the shared recipe at a fixed scalar $\alpha$ | strong simple transfer baseline: isolates what the *meta-learned* initialisation and step sizes add |
| M7 | M5 + the same network, **meta-learned** (FOMAML + Meta-SGD + corruption augmentation), adapted with the shared recipe (headline candidate) | the proposed system |
| M8 | M7 plus CQR/ACI | optional calibration layer (backlog) |

**Headline model (HM)** = the highest rung meeting the promotion rule on the dev window; ties go to the simpler rung.
**Promotion rule.** A component stays if its CRPS improvement over the previous rung has a 95 % block-bootstrap interval excluding 0 at ≥ 2 of 3 horizons and coverage is not worse.

**Ablations.** −fire, −AOD, −ventilation features, −static; GRU vs no calibrator (= M5; TCN only if time allows); FOMAML + Meta-SGD vs fixed $\alpha$; per-tensor vs per-parameter $\boldsymbol\alpha$; corruption augmentation on/off, support-label policy `clean` vs `same_as_inputs`, corruption priors from the co-location audit vs the default ranges; adaptation scope `all` vs head-only (ANIL) vs GRU-only; **A-treebias** (Delhi trees *with* the recent-bias feature, the revision-2 M5: does a meta-learned residual beat simply giving the trees local bias?); with and without the residual bound; Delhi-only vs Delhi+Amritsar+Ludhiana; raw vs log1p target; 7 vs 19 levels; pinball vs Huber/quantile-Huber; RH mask on vs off; **A-warm** (warm-start residual boosting) and **A-blend** (neural/GBM blend) as the plan's alternatives. The revision-2 design (adapted latent → LightGBM stacker) is retired and is not implemented.

**H5** (pre-registered, tiered): **M7 beats M6** on **Delhi season T**, $N\in\{1,2\}$, with ≥ 20 stations — pass if the CRPS-ratio CI upper bound is < 1 and the point estimate ≤ 0.97 (≥ 3 % CRPS). If not, that is the finding. A secondary, descriptive condition **H5-C** repeats it with the corruption prior applied to the available stations' inputs (held-out draws). H1–H4 are tested on Lahore as *indicative* (small effective sample); definitions in [evaluation §1](evaluation-strategy.md#1-evaluation-questions-and-hypotheses).

## 11. Training protocol and reproducibility

**Pipeline order** (each stage consumes only blocks allowed for it; scalers frozen after step 1):

1. **Scalers and thresholds** on block E.
2. **Stage-1 base.** Season-blocked cross-fit on E → **OOF base predictions** (§6); final base on all of E with early stopping on V → **M5**. Tabular **M4** is fit on Lahore A.
3. **Supervised pre-training** of GRU + residual head on E, targets $z-\hat z^{(5)}_{\rm OOF}$ (multi-task, 19 levels × 3 horizons), with corruption augmentation → **M6 initialisation**.
4. **Meta-training** (FOMAML + Meta-SGD, corruption augmentation, §8) on E tasks → **M7 initialisation and learned $\boldsymbol\alpha$**.
5. **Residual-bound threshold.** Simulated deployments on block V with the shared recipe → $\rho_{99}$; threshold pilot for the acceptance rule (2 % / 60 %).
6. **Tuning** on V and the Lahore dev window only.
7. **Calibration** (optional M8).
8. **Bundle**: base boosters (shared), calibrator weights and $\boldsymbol\alpha$ (one bundle each for M6 and M7), scalers, config hash, `base_sha256` binding, latency table, CAMS cycles trained on.
9. **Freeze register** (git tag, config hash, bundle sha256) before the confirmatory window; block T is touched only after it.

* **Splits.** Source: blocks E/V/T as in §7 with a 72-hour embargo and purging of any sample whose lookback or target window overlaps another block or fold (López de Prado, 2018); spatial hold-out of stations inside each block.
* **Determinism.** `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, fixed thread counts, LightGBM `deterministic=true, force_row_wise=true`; the corruption draws use a seeded generator keyed by task id; the manifest records library versions, config hash, data partition hashes (including the OOF fold partition), CAMS cycles in the training window.
* **Tuning.** Seeded random search (30 trials) on the 19-level pinball loss; no extra dependency, no paid tracker. Results are written to `data/processed/runs/<run_id>/` as Parquet and Markdown.
* **Compute (estimates to be measured in Phase 2 and recorded in the model card).** Cross-fitted base: $(K+1)\times57$ small boosters (to be measured, order of minutes to tens of minutes); GRU pre-training: tens of minutes on 2 cores; first-order meta-training ($3{,}000$ outer iterations × 8 tasks × 3 inner steps): on the order of 1–2 h; simulated deployments on V for $\rho_{99}$: hundreds of adaptations of seconds each; the sparsity experiment is sharded by $N$ across matrix jobs (each < 6 h). Revision 2's simulated deployments on block S and the second set of 57 boosters per encoder are no longer needed. All of it is CPU-only and runs on free resources.
* **Fallbacks.** If a bundle fails integrity, compatibility or the base↔calibrator binding, the daily pipeline serves the CAMS-BC baseline (degradation level 2).

## 12. Failure analysis and known risks

| # | Risk | Mitigation / test |
|---|---|---|
| R1 | **Negative transfer.** Delhi is *downwind* of the burn belt; Lahore is *adjacent*, so lag structure and source mix differ | Target-centric features; H4/H5; optional Punjab source cities; always report $N=0$ honestly |
| R2 | **CAMS upgrades shift distributions** (the model cycle changes about yearly; a spring-2026 upgrade was followed by publication delays) | Record cycle; evaluate before/after; retrain trigger |
| R3 | **Right-censoring of sensors** near their effective range (~500 µg m⁻³ typical for optical sensors) biases upper quantiles low | `HIGH_RANGE` flag; verify tails on reference monitors; consider censored pinball |
| R4 | **Train/serve skew** | Same product (CAMS archive), same as-of function and latency table for both; ERA5 never substituted |
| R5 | **Few-shot instability** | Adapt from the fixed meta-initialisation; acceptance rule (≥ 2 % on ≥ 60 % of check days) and residual bound fall back to $\Delta\hat z=0$ (M5) |
| R6 | **Representativeness** (40 km grid vs a point) | Static features, recent-bias input to the residual head, station-level adaptation |
| R7 | **Network non-stationarity** (sensors appear/disappear) | Sparse-network training; $N$ logged per forecast |
| R8 | **Public over-confidence in intervals** | Natural-frequency wording; "8 days in 10 / 1 day in 10"; exceedance probabilities rounded to 5 %; coverage monitoring |
| R9 | **Leakage** via scaling, tuning or overlapping windows | Source-only (block E) scalers; embargo + purging; twin-aware splits; unit tests on as-of joins |
| R10 | **Emission regime change** (policy, kiln technology) | Rolling coverage monitor; retrain trigger |
| R11 | **Provisional-vs-settled truth skew** — provisional API values differ from the archive's settled values, and training labels come from a mix | Train on settled data only; score only first-issuance forecasts against settled truth; report the provisional–settled difference separately |
| R12 | **Residual overshoot on real low-cost support** — a few noisy Lahore days drive the adapted calibrator to large corrections that the clean Delhi tasks never produced | Residual bound $R\le\rho_{99}$ (`rejected_residual_bound`) calibrated on V; check-window acceptance; fallback $\Delta\hat z=0$; low-cost correction; ship M5 if rejections are frequent (X-01) |
| R13 | **Adaptation effect smaller than assumed** — pooled supervised residual learning (M6) may already capture most of the gain, and the corruption augmentation only helps if Lahore's artefacts resemble the prior | M6 trained with the same augmentation (H5 isolates the inner loop); H5-C; head-only/GRU-only scope ablation; `rejected_no_gain` rate reported; fallback rule ships M5 and reports the negative result |
| R14 | **RH-mask tail truncation** — dropping RH > 85 % readings removes foggy, high-PM hours | Mask off by default; audit and RH-stratified ablation before use; mask touches support labels only |
| R15 | **Sparse tail supervision** — all 19 levels are trained, but τ = 0.05/0.95 residuals adapted from ≤ 30 support days rest on about one exceedance | Shared hidden layer across levels; rearrangement; verify tails on reference monitors; report tail pinball separately; optional τ = 0.025/0.975 head extension (owner decision) |
| R16 | **In-sample base predictions** inflate the calibrator's trust in the base and make it under-correct on Lahore | Out-of-fold base predictions only (C1, C2); partition hash in the bundle; test T1-20; experiment X-02 |
| R17 | **Corruption prior mismatch** — Lahore's artefacts differ from the synthetic ranges; support labels may carry the same artefact (not identifiable); reference monitors have none | Priors audited against the co-location fits; support-label policy tested (X-11); `corruption.p` < 1 if needed; H5-C and Lahore stratified by sensor class |

## References

Alduchov & Eskridge (1996) *J. Appl. Meteorol.* — Magnus coefficients. · Bai, Kolter & Koltun (2018) — TCN. · Barkjohn, Gantt & Clements (2021) *Atmos. Meas. Tech.* — PurpleAir correction. · Che et al. (2018) *Sci. Rep.* — GRU-D.
· Chernozhukov, Fernández-Val & Galichon (2010) *Econometrica* — quantile rearrangement. · Cho et al. (2014) — GRU. · Diebold & Mariano (1995) *JBES*. · Finn, Abbeel & Levine (2017) *ICML* — MAML. · Gibbs & Candès (2021) *NeurIPS* — ACI.
· Gneiting & Raftery (2007) *JASA* — proper scoring rules. · Gneiting & Ranjan (2011) *JBES* — threshold-weighted CRPS. · Harvey, Leybourne & Newbold (1997) *Int. J. Forecast.* · Hersbach (2000) *Weather Forecast.* — CRPS. · Ke et al. (2017) *NeurIPS* — LightGBM.
· Koenker & Bassett (1978) *Econometrica* — quantile regression. · Kuleshov, Fenner & Ermon (2018) *ICML*. · Li, Zhou, Chen & Li (2017) — Meta-SGD, arXiv:1707.09835. · López de Prado (2018) *Advances in Financial ML* — purging/embargo. · Matheson & Winkler (1976) *Manage. Sci.* — CRPS.
· Nichol, Achiam & Schulman (2018) — Reptile. · Politis & Romano (1994) *JASA* — stationary bootstrap. · Raghu et al. (2020) *ICLR* — ANIL. · Romano, Patterson & Candès (2019) *NeurIPS* — CQR.
· arXiv:2108.00640 — few-shot calibration of low-cost PM2.5 sensors (cited in the original plan).
