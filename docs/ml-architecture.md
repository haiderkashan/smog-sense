# ML Architecture

> **Audience:** researchers and engineers implementing Phases 2–3 (revision 2). Math renders on GitHub. Every design choice that departs from the original research plan is marked
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
| **Forecast age** | `cams_lead_offset_h` ($\delta = T - B^\*$) | 12 / 18 / 12 / 18 h for $T$ = 00 / 06 / 12 / 18 Z at level 0, **+12 h at degradation level 1**. Training adds a stale-cycle offset $\{0,12,24\}$ h with probabilities $\{0.70,0.20,0.10\}$. An input to every method that uses CAMS covariates (M3b, M5, M6, M7 stackers) |
| **Recent CAMS bias** | trailing 7-day mean of $(x-\mathrm{CAMS})$ at the station | lets the model act as a local post-processor (representativeness: 40 km grid vs a point) |
| **Fire** | ring counts/FRP, $E_{24},E_{48},E_{72}$ | §3 |
| **Calendar** | sin/cos of local hour, weekday, day-of-year; event flags | Diwali flag exists for Indian domains only and is *dropped* (not zero-filled) when adapting to Lahore |
| **Static** | lat, lon, elevation, distance to centre (optional built-up fraction, road density) | so unseen stations are not anonymous |

### 2.3 Lagged cross-correlation (diagnostic, not a leakage vector)

To check that the fire kernel's time scales are physically plausible, compute on *deseasonalised* series from the **training seasons only**

$$
\rho(\ell)=\frac{\sum_t\big(E_{t-\ell}-\bar E\big)\big(\tilde x_t-\bar x\big)}{\sqrt{\sum_t(E_{t-\ell}-\bar E)^2}\sqrt{\sum_t(\tilde x_t-\bar x)^2}},\qquad \ell=0,\dots,96\ \mathrm h,
$$

with stationary-bootstrap confidence bands. The maximising lag $\ell^\*$ should agree with the typical advective time $d/(3.6\,\bar w)$ ($\approx 8$–$40$ h for $100$–$300$ km at $2$–$4$ m s⁻¹). The same analysis is run for upstream CAMS PM2.5. Selected lags are frozen before touching validation or test data.

### 2.4 Tensors, masks and scaling

The encoder input is $X\in\mathbb R^{L\times C}$, $L=72$, with the ten channels in `configs/features.yaml` plus a mask and a time-since-observation channel for PM2.5 (inputs $[x\odot m,\ m,\ \delta_t]$, after Che et al., 2018; $\delta_t$ here is the time since the last observation, not the forecast age).
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

**GRU** (Cho et al., 2014), two layers, hidden 64:

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z),\quad r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r),\quad
\tilde h_t=\tanh\!\big(W_hx_t+U_h(r_t\odot h_{t-1})+b_h\big),\quad
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t ,
$$

followed by a linear map of $h_L$ to a latent $\mathbf z\in\mathbb R^{32}$ (the *latent* that the stacker consumes, §6). The GRU has ≈ 42 k parameters (two layers: ≈ 15 k + ≈ 25 k, plus the 64→32 projection; a unit test asserts the count), so the whole neural stack is ≈ 65 k and small enough to meta-train on CPU.
**TCN** (Bai et al., 2018) is an optional alternative configuration (cut from the planned scope, see the roadmap): causal dilated convolutions $(F\!*\!x)(t)=\sum_{k=0}^{K-1}f(k)\,x_{t-dk}$ in residual blocks. With kernel $K=3$, $n_c=2$ convolutions per block and dilations $\{1,2,4,8,16\}$ the receptive field is

$$
R=1+n_c\,(K-1)\sum_i d_i=1+2\cdot2\cdot31=125\ \mathrm h\ \ge\ L=72 .
$$

The encoder is a configuration switch; the ladder (§10) decides whether a sequence encoder earns its place over purely tabular lags (M5).

## 5. Quantile loss, the temporary neural head, and the quantile function

**Pinball (quantile) loss** (Koenker & Bassett, 1978): $\rho_\tau(u)=u\,(\tau-\mathbf 1\{u<0\})=\max\{\tau u,(\tau-1)u\}$. For $Y\sim F$:
$\frac{d}{dq}\mathbb E\,\rho_\tau(Y-q)=F(q)-\tau$, so the minimiser is $F^{-1}(\tau)$. The penalty ratio for under- vs over-prediction is $\tau/(1-\tau)$ — $9{:}1$ at $\tau=0.9$, as the plan notes.

**Temporary neural head — non-crossing by construction, 3 levels × 3 horizons (Δ).** The neural network is used for *representation learning and adaptation*, not as the product's output layer. Its head predicts only $\tau\in\{0.10,0.50,0.90\}$ for each horizon: $\hat z_{1}=a_1,\ \hat z_{2}=\hat z_{1}+\operatorname{softplus}(a_2),\ \hat z_{3}=\hat z_{2}+\operatorname{softplus}(a_3)$. Loss over a batch $B$ with validity masks $\mu_{s,T,h}$ and $\mathcal Q_3=\{0.10,0.50,0.90\}$:

$$
\mathcal L(\theta)=\frac1{\sum\mu}\sum_{(s,T,h)\in B}\mu_{s,T,h}\;\frac1{3}\sum_{\tau\in\mathcal Q_3}\rho_{\tau}\big(z_{s,T,h}-\hat z_{\tau}(s,T,h)\big).
$$

The head is **temporary**: it exists in meta-training, in the inner adaptation loop and in the acceptance check (§8), and it is *not* part of the published forecast. The 19-level product comes from the stacker (§6).
**Untrained-tail note.** Because the neural loss never sees $\tau\notin\mathcal Q_3$, the latent is not directly optimised for the 0.05/0.95 tails; tail information reaches the stacker through the tabular inputs (fire, ventilation, recent bias) and whatever the latent carries incidentally. This is a known limitation (risk R15). An optional extension of the head grid to $\tau=0.025/0.975$ (and the corresponding scoring grid) is an owner decision, not applied.

**Quantile function object.** Interior: piecewise-linear in $\tau$ through the 19 knots. Tails: upper exponential, $q(\tau)=q_K+s_{hi}[\Lambda(\tau)-\Lambda(\tau_K)]$ with $\Lambda(\tau)=-\ln(1-\tau)$ and
$s_{hi}=\frac{q_K-q_{K-1}}{\Lambda(\tau_K)-\Lambda(\tau_{K-1})}$; lower, log-linear $q(\tau)=\max\{0,\,q_1+s_{lo}[\ln\tau-\ln\tau_1]\}$. This object provides `ppf`, `cdf`, $P(Y>\theta)$, $P(\text{category})$ and exact-integral CRPS.
**Measured:** against analytic CRPS of lognormal ($\sigma\in\{.3,.5,.8\}$) and gamma distributions, the estimator's relative error is **0.2 % mean (≤ 0.5 % max) at $K=19$ but 1.0–3.5 % mean (up to 6.2 % max) at $K=7$**, and 13 % for a naïve 7-level estimate without a tail model. That is why the scoring grid has 19 levels.

**Monotone rearrangement.** Independently trained quantiles (the 57 boosters of §6) can cross (in a synthetic test, 7 of 500 rows). Sorting the $K$ values restores monotonicity and **never increases the total pinball loss**: $f(\tau,q)=\rho_\tau(y-q)$ has $\partial^2f/\partial\tau\,\partial q=-1<0$ (submodular), so by the rearrangement inequality $\sum_kf(\tau_k,q_{\pi(k)})$ is minimised by the sorted pairing (cf. Chernozhukov et al., 2010).

## 6. LightGBM quantile ensemble and the decoupled stacker

**Δ — why decoupled.** Meta-learning (MAML) needs differentiable parameters; gradient-boosted trees are not. The design therefore *separates the two jobs*: MAML acts only on the neural encoder; after adaptation the encoder is **frozen**, its 32-d latent is extracted, and an *independent* LightGBM stacker maps $[\text{latent}\,\|\,\text{tabular}\,\|\,\delta]$ to quantiles. Trees do what they are good at — abrupt, non-linear interactions such as "high upwind fire matters only when the wind aligns **and** the boundary layer is shallow".

**No native multi-quantile in LightGBM.** The plan states that LightGBM "natively supports multi-quantile regression" through an `alpha` array. It does not: `alpha` is a single `double`; passing `[0.1, 0.5, 0.9]` raises *"Parameter alpha should be of type double"* (tested, LightGBM 4.7). SmogSense trains $3\times19=57$ boosters with `objective=quantile`, `alpha=<τ>`, then applies rearrangement (§5).

**Inputs.** $[\,\mathbf z\ (32)\ \|\ \text{block covariates}\ \|\ \text{fire}\ \|\ \text{recent bias}\ \|\ \text{calendar}\ \|\ \text{static}\ \|\ \text{cams\_lead\_offset\_h}\,]$.

**Regularisation for tails.** A leaf should hold at least two expected tail observations: $n_{\rm leaf}\cdot\min(\tau,1-\tau)\ge2\Rightarrow$ `min_data_in_leaf` $\ge40$ at $\tau=0.05$ (the default).

**Stacker rows come from simulated deployments (C1, C2).** The stacker is trained on rows that look like what it will see in deployment: *adapted* latents on stations the encoder has not been fitted to. For each simulated deployment (a sparse-network task drawn from block S, §7): draw $N$, the available set $A$ and a support window; run the **exact deployment recipe** (§8) from the meta-initialisation; freeze; extract latents on the later query window of the held-out stations $B$ (after the 72-h embargo); store one row $[\mathbf z,\ \text{tabular},\ \delta,\ y]$ with its task id. Early stopping uses block V with grouping by task so that rows of one task never straddle train and validation.
**No cross-fitting (K-fold).** Independently trained encoders have incoherent latent frames (latent axis 7 means different things in two runs), so rows from different encoders cannot be pooled. One final encoder, trained on block E only, produces all rows.

**Binding (C3).** The model bundle stores `encoder_sha256` in the stacker manifest; loading refuses a mismatched pair (degradation level 2). **M6 and M7 each own their stacker**, trained on rows from their own encoder; **M5 is tabular** (no encoder).
**No warm-start residual boosting and no neural/GBM blend in the product path.** Both are kept as ablations (A-warm, A-blend) because the plan proposed them.

**Fallback rule.** If the stacker is not robust to latent shift (it fails the drift/acceptance checks in simulation or on the Lahore dev window) or adaptation adds nothing over the unadapted encoder, **ship M5 as the headline and report the negative result** with the same prominence as a positive one.

## 7. Sparse-network task construction

**Δ — the plan conflates two different "N"s.** In the plan's sensor-sparsity experiment, "zero local stations" is described as pure zero-shot transfer, but the encoder consumes the *target station's PM2.5 history*; with no local station there is no history to consume. The consistent definition: $N$ is the number of **available stations**, which supply *both* the labels used for adaptation *and* the observed-history input at inference.

**Source-domain season blocks (E/S/V/T).** The source seasons (Delhi) are split chronologically into four non-overlapping blocks, each with at least one season, with a 72-h embargo and purging at every boundary:

| Block | Role | Example, 8 seasons |
|---|---|---|
| **E** | encoder: scalers, supervised pre-training, meta-training | oldest 4 |
| **S** | stacker rows from simulated deployments (out-of-sample for the encoder) | next 2 |
| **V** | tuning, early stopping, drift-threshold calibration | 1 |
| **T** | Delhi confirmatory test (H5), untouched until the freeze | newest 1 |

**Task.** A **task** $\mathcal T=(d,p,A,B,W^{\rm sup},W^{\rm qry})$: domain $d$; season $p$ within one block; available set $A\subset\mathcal S_d$ with $|A|=N\sim\mathrm{Unif}\{0,1,2,3,5,8\}$; held-out set $B=\mathcal S_d\setminus A$ (capped for speed); an earlier support window and a later query window separated by a 72-hour embargo.
**Twin-aware:** stations in the same `colocated_group` always fall on the same side of the $A/B$ split (a held-out twin of an available station would leak its readings). Station identity is `sensor_id`.

* **History input.** For any target $s$: $H_A(s,t)=\sum_{a\in A}w_a(s)\,x_{a,t}\big/\sum_aw_a(s)$ with inverse-distance weights $w_a(s)\propto\mathrm{dist}(s,a)^{-2}$ within 25 km (plus a "distance to nearest available" feature). If $N=0$ the channel is fully masked: **sensor-free mode**, driven by CAMS, meteorology and fire.
* **Two evaluation modes.** *Own-history* (forecast the future at stations in $A$) and *neighbour-history* (forecast at $B$ using $A$ — i.e. at places without a sensor). Training samples draw "own history present" with probability 0.5. The sparsity curve's primary mode is **leave-stations-out** (neighbour-history).
* **Modality and group dropout.** With probability 0.3 the whole PM2.5 history is masked (so sensor-free mode is trained, not merely tested); fire and AOD groups are each dropped with probability 0.1 so a failed FIRMS call degrades gracefully.
* **Baselines use the available set only.** M0, M2, M3 and M3b see only $A$ (support window and history). At $N=0$ only M1 (deterministic) and the sensor-free learned methods are defined.
* **Explicit $N=0$ behaviour (C7).** With no support there is nothing to adapt on: the encoder stays at its meta-initialisation, the stacker is unchanged, the degradation level is 0 and `adaptation_status = skipped_n0`. If $N\ge1$ but the support window has too few valid days (default: fewer than 5, `adaptation.min_support_days`, to be fixed by the threshold pilot), the same fallback applies with `skipped_insufficient_support`.

Meta-training tasks come from every Delhi station-season in block E (≈ hundreds). Optional extra source domains (Amritsar, Ludhiana) add task diversity if Delhi-only transfer is brittle.

## 8. Meta-learning (MAML) and adaptation

**Setting.** Neural parameters $\theta=(\theta_{\rm enc},\theta_{\rm head})$ where the head is the *temporary* 3-level × 3-horizon non-crossing head of §5. Task loss $\mathcal L_{\mathcal T}(\theta;D)=\frac1{|D|}\sum_{(x,y)\in D}\ell\big(f_\theta(x),y\big)$ with $\ell$ the masked mean pinball loss over the three levels and three horizons.

**Loss and clipping (C8).** The inner and outer loss is **pinball**; gradient-norm clipping at **1.0** in both loops. Huber / quantile-Huber is an **ablation**, not the default: pinball is the proper score for the target, and heavy-tail robustness is better handled by the clip.

**Inner loop** (support set $D_i^{\rm sup}$, $J$ steps, rate $\alpha$; all parameters adapted, `adapt_scope: all`):

$$
\theta_i^{(0)}=\theta,\qquad \theta_i^{(j+1)}=\theta_i^{(j)}-\alpha\,\nabla_{\theta_i^{(j)}}\mathcal L_{\mathcal T_i}\big(\theta_i^{(j)};D_i^{\rm sup}\big),\qquad \theta_i'=\theta_i^{(J)} .
$$

**Outer loop** (query set $D_i^{\rm qry}$, later in time than the support set):

$$
\min_\theta\ \sum_{\mathcal T_i\sim p(\mathcal T)}\mathcal L_{\mathcal T_i}\big(\theta_i';D_i^{\rm qry}\big),\qquad
\theta\leftarrow\theta-\beta\,\nabla_\theta\sum_i\mathcal L_{\mathcal T_i}\big(\theta_i';D_i^{\rm qry}\big)\ \ (\text{Adam}).
$$

**Gradient structure** (Finn et al., 2017). With $H_i^{(j)}=\nabla^2_{\theta}\mathcal L_{\mathcal T_i}(\theta_i^{(j)};D_i^{\rm sup})$,

$$
\nabla_\theta\,\mathcal L^{\rm qry}_i(\theta_i')=\Big[\prod_{j=0}^{J-1}\big(I-\alpha H_i^{(j)}\big)\Big]\nabla_{\theta_i'}\mathcal L^{\rm qry}_i(\theta_i') .
$$

The plan's "gradient of the gradient" is exactly the Hessian product. **Algorithms** (config `meta.algorithm`): **FOMAML** (default) sets $H\approx0$. *Reptile* ($\theta\leftarrow\theta+\epsilon\,\frac1n\sum_i(\theta_i^{(J)}-\theta)$, Nichol et al., 2018) and *second-order MAML* ($J\le3$) remain implementable but are **cut from the planned scope** and run only if time allows.
**Default: first-order, $J=3$, $\alpha=0.01$**, on CPU via `torch.func.functional_call`.
**ANIL is a negative control only (C4).** ANIL (Raghu et al., 2020) adapts only the head in the inner loop. In the decoupled design the stacker consumes the *encoder latent*; head-only adaptation leaves the latent unchanged, so it should deliver **no** gain over the unadapted encoder. It is run once as a control: if ANIL matched FOMAML, the gain would not be coming from latent adaptation and the design's premise would fail.

**One shared recipe (C5).** A single config block `adaptation.inner` (loss, $\alpha$, $J$, support window, masks, clip, `adapt_scope`) is used by meta-training, simulated-deployment (S rows), and deployment. The support window is sampled from {7, 14, 30} days in meta-training and deployment uses the same set. **Deployment $J$ equals meta-training $J$**; any other $J$ is an ablation. Adaptation always starts from the *meta-initialisation* (never from yesterday's adapted weights), so it is a deterministic function of the data and cannot drift.

**Deployment adaptation (Lahore) with acceptance and drift guard (C6).**
1. Take the available stations' support window; split **chronologically**: *fit* part (earlier), a **72-h embargo**, and *check* part (latest).
2. Adapt on the fit part with the shared recipe.
3. Compare adapted vs meta-initial **check pinball** loss. **Accept iff** the improvement is **≥ 2 %** *and* the adapted model is better on **≥ 60 %** of check days, **and** the latent-drift statistic $D\le$ the **99th percentile of $D$ over the simulated S tasks**.
4. $D=\frac1{32}\sum_{k=1}^{32}\big|\mu^{\rm target}_k-\mu^{\rm ref}_k\big|/\sigma^{\rm ref}_k$, where $\mu^{\rm target}$ is the mean adapted latent on the target's recent window and $(\mu^{\rm ref},\sigma^{\rm ref})$ are the mean and standard deviation of the S-task latents. The threshold is calibrated on block S/V, never on Lahore data.
5. If accepted, the adapted encoder is re-run on the full support window for the final latents. If rejected, fall back to the meta-initialisation latents; the stacker is unchanged.

**`adaptation_status`** (logged in `forecast_log` and the run manifest) takes exactly one of: `adapted`, `skipped_n0`, `skipped_insufficient_support`, `rejected_no_gain`, `rejected_latent_drift`. A rejection is a normal outcome, not a failure, and never changes the degradation level.

**Expected effect on H5.** Honest expectation: modest. The stacker already sees the recent-bias and CAMS covariates and can absorb part of the domain shift on its own, which shrinks the room left for meta-learning; the literature gives real reasons to expect plain fine-tuning (M6) to be competitive. H5 therefore tests M7 against M6, and a null result is a finding.

**Memory hygiene.** The adaptation stage must fit the 7 GB runner with margin: it copies *parameters* (not the model) per task, runs under `torch.no_grad` wherever gradients are not needed, caps the support set deterministically (≤ 30 days × available stations, subsampled by a fixed seed if larger), releases tensors and calls `gc.collect()` between tasks, and records its **peak RSS** in the run manifest. A test asserts a peak-memory bound on a synthetic worst-case support set.

**Optional RH mask on support readings (C9; off by default).** When enabled (`adaptation.rh_mask.enabled: true`): drop *support* readings from low-cost sensors whose co-measured RH (device RH if present, otherwise CAMS RH) exceeds **85 %**, provided at least **50 %** of the support window survives; otherwise skip the mask. The mask applies to the *labels used for adaptation only* — never to targets, check windows or evaluation rows. Rationale: optical readings inflate under hygroscopic growth. Risk (R14): high RH coincides with fog and the highest PM, so masking can truncate exactly the upper tail. It is therefore **off** until the mask audit and ablation (RH-stratified scoring: none / 75–85 / above 85) show a benefit.

**Conditions of the decoupled design (C1–C9).**

| ID | Condition | Where |
|---|---|---|
| C1 | Stacker rows use **adapted** latents from simulated deployments (not the meta-init latents) | §6 |
| C2 | Rows come from an **out-of-sample season block S** (not the encoder's training seasons) | §6, §7 |
| C3 | Stacker is **bound** to its encoder by `encoder_sha256`; mismatch ⇒ level 2 | §6 |
| C4 | ANIL is a **negative control only** | §8 |
| C5 | **One shared recipe** block; meta-training $J$ = deployment $J$ | §8 |
| C6 | **Fit / embargo / check** split plus the **latent-drift guard** | §8 |
| C7 | **Explicit $N=0$** and insufficient-support behaviour | §7 |
| C8 | **Pinball** with gradient clip 1.0; Huber is an ablation | §8 |
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
| M5 | Tabular LightGBM trained on Delhi, no weight adaptation; features computed from **A** (feature-based adaptation) | pure transfer; also the **fallback headline** (§6) |
| M6 | Supervised-pretrained encoder + same adaptation recipe + **own stacker** | strong simple transfer baseline |
| M7 | Meta-learned encoder + same recipe + **own stacker** (headline candidate) | the proposed system |
| M8 | M7 plus CQR/ACI | optional calibration layer (backlog) |

**Headline model (HM)** = the highest rung meeting the promotion rule on the dev window; ties go to the simpler rung.
**Promotion rule.** A component stays if its CRPS improvement over the previous rung has a 95 % block-bootstrap interval excluding 0 at ≥ 2 of 3 horizons and coverage is not worse.

**Ablations.** −fire, −AOD, −ventilation features, −recent bias, −static; GRU vs no encoder (tabular lags only; TCN only if time allows); FOMAML vs the **ANIL control**; Delhi-only vs Delhi+Amritsar+Ludhiana; raw vs log1p target; 7 vs 19 levels; with and without modality dropout;
**A-warm** (warm-start residual boosting) and **A-blend** (neural/GBM blend) as the plan's alternatives; pinball vs Huber/quantile-Huber; RH mask on vs off; stacker vs neural-only head (3 levels); with and without the drift guard.

**H5** (pre-registered, tiered): **M7 beats M6** on **Delhi season T**, $N\in\{1,2\}$, with ≥ 20 stations — pass if the CRPS-ratio CI upper bound is < 1 and the point estimate ≤ 0.97 (≥ 3 % CRPS). If not, that is the finding. H1–H4 are tested on Lahore as *indicative* (small effective sample); definitions in [evaluation §1](evaluation-strategy.md#1-evaluation-questions-and-hypotheses).

## 11. Training protocol and reproducibility

**Pipeline order** (each stage consumes only blocks allowed for it; scalers frozen after step 1):

1. **Scalers and thresholds** on block E.
2. **Supervised pre-training** of the encoder (multi-task, 3 levels × 3 horizons) on E → M6 encoder.
3. **Meta-training** (FOMAML) on E tasks → M7 encoder.
4. **Simulated deployments on S** with the shared recipe → stacker rows (one set per encoder), and the **drift threshold** (99th percentile of $D$).
5. **Stacker fit** (57 boosters per encoder) with early stopping on V; **tabular M4/M5** fit on E+S (M5) / Lahore A (M4).
6. **Tuning** on V and the Lahore dev window only; threshold pilot (acceptance 2 % / 60 %, $D$ quantile).
7. **Calibration** (optional M8).
8. **Bundle**: encoder weights, stacker boosters, scalers, config hash, `encoder_sha256` binding, latency table, CAMS cycles trained on.
9. **Freeze register** (git tag, config hash, bundle sha256) before the confirmatory window; block T is touched only after it.

* **Splits.** Source: blocks E/S/V/T as in §7 with a 72-hour embargo and purging of any sample whose lookback or target window overlaps another block (López de Prado, 2018); spatial hold-out of stations inside each block.
* **Determinism.** `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, fixed thread counts, LightGBM `deterministic=true, force_row_wise=true`; the manifest records library versions, config hash, data partition hashes, CAMS cycles in the training window.
* **Tuning.** Seeded random search (30 trials) on $\tau\in\{.1,.5,.9\}$; no extra dependency, no paid tracker. Results are written to `data/processed/runs/<run_id>/` as Parquet and Markdown.
* **Compute (estimates to be measured in Phase 2 and recorded in the model card).** GRU pre-training: tens of minutes on 2 cores; first-order meta-training ($3{,}000$ outer iterations × 8 tasks × 3 inner steps): on the order of 1–2 h; simulated deployments on S and the 57-booster fits: to be measured (~tens of minutes each); the sparsity experiment is sharded by $N$ across matrix jobs (each < 6 h). All of it is CPU-only and runs on free resources.
* **Fallbacks.** If a bundle fails integrity, compatibility or the encoder↔stacker binding, the daily pipeline serves the CAMS-BC baseline (degradation level 2).

## 12. Failure analysis and known risks

| # | Risk | Mitigation / test |
|---|---|---|
| R1 | **Negative transfer.** Delhi is *downwind* of the burn belt; Lahore is *adjacent*, so lag structure and source mix differ | Target-centric features; H4/H5; optional Punjab source cities; always report $N=0$ honestly |
| R2 | **CAMS upgrades shift distributions** (the model cycle changes about yearly; a spring-2026 upgrade was followed by publication delays) | Record cycle; evaluate before/after; retrain trigger |
| R3 | **Right-censoring of sensors** near their effective range (~500 µg m⁻³ typical for optical sensors) biases upper quantiles low | `HIGH_RANGE` flag; verify tails on reference monitors; consider censored pinball |
| R4 | **Train/serve skew** | Same product (CAMS archive), same as-of function and latency table for both; ERA5 never substituted |
| R5 | **Few-shot instability** | Adapt from the fixed meta-initialisation; acceptance rule (≥ 2 % on ≥ 60 % of check days) falls back to the meta-initial latents |
| R6 | **Representativeness** (40 km grid vs a point) | Static features, recent-bias feature, station-level adaptation |
| R7 | **Network non-stationarity** (sensors appear/disappear) | Sparse-network training; $N$ logged per forecast |
| R8 | **Public over-confidence in intervals** | Natural-frequency wording; "8 days in 10 / 1 day in 10"; exceedance probabilities rounded to 5 %; coverage monitoring |
| R9 | **Leakage** via scaling, tuning or overlapping windows | Source-only (block E) scalers; embargo + purging; twin-aware splits; unit tests on as-of joins |
| R10 | **Emission regime change** (policy, kiln technology) | Rolling coverage monitor; retrain trigger |
| R11 | **Provisional-vs-settled truth skew** — provisional API values differ from the archive's settled values, and training labels come from a mix | Train on settled data only; score only first-issuance forecasts against settled truth; report the provisional–settled difference separately |
| R12 | **Latent drift on real low-cost support** — S-task latents come from regulatory-grade Delhi stations, Lahore support comes from noisy low-cost sensors, so real latents can leave the stacker's training region | Drift statistic $D$ with a 99th-percentile guard (`rejected_latent_drift`); low-cost correction; fallback to meta-initial latents; ship M5 if drift rejections are frequent |
| R13 | **Adaptation effect smaller than assumed** — the stacker absorbs the domain shift via recent-bias features, leaving little for MAML | M6 comparison (H5); ANIL control; `rejected_no_gain` rate reported; fallback rule ships M5 and reports the negative result |
| R14 | **RH-mask tail truncation** — dropping RH > 85 % readings removes foggy, high-PM hours | Mask off by default; audit and RH-stratified ablation before use; mask touches support labels only |
| R15 | **Untrained tails** — the neural loss never sees $\tau=0.05, 0.95$, so the latent may carry little tail information | Tail-focused tabular inputs; verify tails on reference monitors; optional $\tau=0.025/0.975$ head extension (owner decision) |

## References

Alduchov & Eskridge (1996) *J. Appl. Meteorol.* — Magnus coefficients. · Bai, Kolter & Koltun (2018) — TCN. · Barkjohn, Gantt & Clements (2021) *Atmos. Meas. Tech.* — PurpleAir correction. · Che et al. (2018) *Sci. Rep.* — GRU-D.
· Chernozhukov, Fernández-Val & Galichon (2010) *Econometrica* — quantile rearrangement. · Cho et al. (2014) — GRU. · Diebold & Mariano (1995) *JBES*. · Finn, Abbeel & Levine (2017) *ICML* — MAML. · Gibbs & Candès (2021) *NeurIPS* — ACI.
· Gneiting & Raftery (2007) *JASA* — proper scoring rules. · Gneiting & Ranjan (2011) *JBES* — threshold-weighted CRPS. · Harvey, Leybourne & Newbold (1997) *Int. J. Forecast.* · Hersbach (2000) *Weather Forecast.* — CRPS. · Ke et al. (2017) *NeurIPS* — LightGBM.
· Koenker & Bassett (1978) *Econometrica* — quantile regression. · Kuleshov, Fenner & Ermon (2018) *ICML*. · López de Prado (2018) *Advances in Financial ML* — purging/embargo. · Matheson & Winkler (1976) *Manage. Sci.* — CRPS.
· Nichol, Achiam & Schulman (2018) — Reptile. · Politis & Romano (1994) *JASA* — stationary bootstrap. · Raghu et al. (2020) *ICLR* — ANIL. · Romano, Patterson & Candès (2019) *NeurIPS* — CQR.
· arXiv:2108.00640 — few-shot calibration of low-cost PM2.5 sensors (cited in the original plan).
