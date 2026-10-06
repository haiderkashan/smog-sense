# ML Architecture

> **Audience:** researchers and engineers implementing Phases 2–3. Math renders on GitHub. Every design choice that departs from the original research plan is marked
> **Δ** and justified in the [verification log](verification-log.md).

## 1. Problem formulation

**Notation.** Stations $s\in\mathcal S_d$ belong to a domain $d$ (target: Lahore; source: Delhi). Issuance times $T$ lie on a UTC lattice (training: 00/06/12/18 h; operations: 00 h). $x_{s,t}$ is quality-controlled hourly PM2.5 (µg m⁻³)
with validity $m_{s,t}\in\{0,1\}$ ($1$ = valid and *not* imputed). Horizons $h\in\mathcal H=\{24,48,72\}$.

**Target (Δ).** The original plan forecasts the concentration "at time $T+h$". Hourly low-cost readings are noisy, AQI categories are defined on 24-hour means, and CAMS is a 40 km model whose hourly skill is limited, so SmogSense forecasts the **24-hour block mean**:

$$
y_{s,T,h}=\frac1{|V|}\sum_{t\in V}x_{s,t},\qquad V=\{\,t\in[T+h-24\,\mathrm h,\;T+h):m_{s,t}=1\,\},\qquad y_{s,T,h}\ \text{defined iff}\ |V|\ge 18 .
$$

(Hour-start labels; the block is the 24 hours $T+h-24,\dots,T+h-1$. The $T+24,T+48,T+72$ of the plan are the block *ends*.)
The city value is a pseudo-location with target = mean of group-collapsed station block means over a panel frozen at season freeze, requiring ≥3 valid stations. Its quantiles are modelled directly, not composed from station quantiles. The bulletin states panel size and reference/low-cost split.

**Probabilistic output.** For each $(s,T,h)$ the model returns quantiles $\hat q_{s,T,h}(\tau)$ for $\tau\in\mathcal Q=\{0.05,0.10,\dots,0.95\}$ ($K=19$). The **public** levels are $0.10,0.50,0.90$: an **80 % central interval** $[\hat q_{.10},\hat q_{.90}]$ (not 90 %) and a
median. $\hat q_{.90}$ is *not* a worst case; about one day in ten exceeds it (§[Dissemination](dissemination-and-ui.md#2-risk-communication-rules-for-probabilistic-forecasts)).

**Transform.** Models are trained on $z=\log(1+y)$. Quantiles commute with monotone maps, $Q_\tau[g(Y)]=g(Q_\tau[Y])$, so $\hat q=\exp(\hat z)-1$ estimates the same population quantile while stabilising optimisation of the heavy right tail. All scoring is in µg m⁻³.

**Inputs available at $T$** (as-of rule, [data engineering §11](data-engineering.md#11-temporal-alignment)): observation history from the *available* stations $A$ ($|A|=N$, possibly $0$); CAMS stitched lookback and the forecast covariates for the three blocks; fire exposure; static station descriptors; calendar.

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
| | `cams_lead_offset_h` (δ = T − B*) | plus stale-cycle augmentation in training: extra offset {0, 12, 24} h with probabilities {0.70, 0.20, 0.10} |
| | `cams_lead_offset_h` ($\delta = T - B^*$) | plus stale-cycle augmentation in training: extra offset {0, 12, 24} h with probabilities {0.70, 0.20, 0.10} |
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

The encoder input is $X\in\mathbb R^{L\times C}$, $L=72$, with the ten channels in `configs/features.yaml` plus a mask and a time-since-observation channel for PM2.5 (inputs $[x\odot m,\ m,\ \delta]$, after Che et al., 2018).
Scalers (robust, median/IQR) are fit on the **source-domain training split only** and then frozen: re-fitting them on Lahore would silently erase the very domain shift that adaptation must learn, and would change the meaning of inner-loop gradients.

## 3. Fire transport exposure

**Δ — target-centric, wind-aligned; not a fixed upwind box.** The plan places a box "upwind of Lahore" using north-westerly winds from Indian Punjab. That geometry suits Delhi (downwind of the burn belt) but not Lahore: Indian Punjab lies **east and south-east** of Lahore (Amritsar ≈ 50 km east), so north-westerly flow brings Pakistani-Punjab air, and Indian-Punjab smoke arrives under easterly/south-easterly flow.
Rather than hard-coding a direction, define exposure relative to each target so the same definition transfers between cities and the model discovers which sectors matter.

For detection $i$ with FRP $f_i$ (MW), platform weight $\omega_i=1/n_{\text{plat}}(\tau_i)$, acquisition time $\tau_i$, distance $d_i$ (km) and $\beta$ = bearing of the vector fire→target. $\phi = \operatorname{atan2}(u, v)$, the direction the air moves toward, clockwise from north. $a_i(t) = \max(0, \cos(\phi(t) - \beta_i))$. Target wind speed $w(t)=\max(\lVert\mathbf u(t)\rVert,w_{\min})$:

Lahore 31.5204°N, 74.3587°E; a fire at Amritsar 31.634°N, 74.8723°E. Offset: ΔE ≈ 48.8 km, ΔN ≈ 12.6 km, so target→fire azimuth ≈ 75.5° clockwise from north. Fire→target β ≈ 255.5°. Wind transport direction φ = atan2(u, v), the direction air moves toward. Easterly wind (from the east, u = −5): φ = 270°, so a = cos(14.5°) ≈ 0.97 (high exposure). Westerly wind (u = +5): φ = 90°, so cos(−165.5°) ≈ −0.97, clipped to 0.

$$
a_i(t)=\max\{0,\cos(\phi(t)-\beta_i)\},\qquad
\Theta_i(t)=\frac{d_i}{3.6\,w(t)}\ \ [\mathrm h],\qquad
K_i(t)=a_i(t)\,e^{-\Theta_i(t)/\tau_{\rm decay}}\;e^{-(\min(t,T)-\tau_i)^{+}/\tau_{\rm age}},
$$

$$
E(t)=\sum_{i:\ \tau_i\le T}\omega_i\,f_i\,K_i(t),\qquad
E_h=\frac1{24}\sum_{t\in[T+h-24,\,T+h)}E(t).
$$

For $t>T$ the wind comes from the **CAMS forecast** and the fire set is frozen at the detections knowable at $T$ (persistence of the recent burning rate). Defaults: $\tau_{\rm decay}=24$ h, $\tau_{\rm age}=48$ h, $w_{\min}=0.5$ m s⁻¹, chosen on source seasons and then frozen.
The compact feature set is the four ring counts, four ring FRP sums and $E_{24},E_{48},E_{72}$; the full set adds ring × sector counts. **Interpretability output:** mean |SHAP| by sector, per domain — the expected Lahore signature is easterly/south-easterly weight in October–November; its absence would be a red flag worth investigating.
**Sensor continuity:** all counts and FRP are platform-normalised so that the Suomi-NPP retirement (1 Nov 2026) and the MODIS decline do not create artificial trends.

## 4. Sequence encoder

**GRU** (Cho et al., 2014), two layers, hidden 64:

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z),\quad r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r),\quad
\tilde h_t=\tanh\!\big(W_hx_t+U_h(r_t\odot h_{t-1})+b_h\big),\quad
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t ,
$$

followed by a linear map of $h_L$ to a latent $z\in\mathbb R^{32}$. **TCN** (Bai et al., 2018): causal dilated convolutions $(F\!*\!x)(t)=\sum_{k=0}^{K-1}f(k)\,x_{t-dk}$ in residual blocks. With kernel $K=3$, $n_c=2$ convolutions per block and dilations $\{1,2,4,8,16\}$ the receptive field is

$$
R=1+n_c\,(K-1)\sum_i d_i=1+2\cdot2\cdot31=125\ \mathrm h\ \ge\ L=72 .
$$

Both encoders have ≈ 40 k parameters (the whole neural stack ≈ 65 k), small enough to meta-train on CPU. The encoder is a configuration switch; the ladder (§10) decides whether a sequence encoder earns its place over purely tabular lags.

## 5. Quantile heads and distributional output

**Pinball (quantile) loss** (Koenker & Bassett, 1978): $\rho_\tau(u)=u\,(\tau-\mathbf 1\{u<0\})=\max\{\tau u,(\tau-1)u\}$. For $Y\sim F$:
$\frac{d}{dq}\mathbb E\,\rho_\tau(Y-q)=F(q)-\tau$, so the minimiser is $F^{-1}(\tau)$. The penalty ratio for under- vs over-prediction is $\tau/(1-\tau)$ — $9{:}1$ at $\tau=0.9$, as the plan notes.

**Neural head — non-crossing by construction.** For each horizon: $\hat z_1=a_1,\ \hat z_k=\hat z_{k-1}+\operatorname{softplus}(a_k)$, $k=2..K$. Loss over a batch $B$ with validity masks $\mu_{s,T,h}$:

$$
\mathcal L(\theta)=\frac1{\sum\mu}\sum_{(s,T,h)\in B}\mu_{s,T,h}\;\frac1K\sum_{k=1}^{K}\rho_{\tau_k}\big(z_{s,T,h}-\hat z_{k}(s,T,h)\big).
$$

**Quantile function object.** Interior: piecewise-linear in $\tau$ through the 19 knots. Tails: upper exponential, $q(\tau)=q_K+s_{hi}[\Lambda(\tau)-\Lambda(\tau_K)]$ with $\Lambda(\tau)=-\ln(1-\tau)$ and
$s_{hi}=\frac{q_K-q_{K-1}}{\Lambda(\tau_K)-\Lambda(\tau_{K-1})}$; lower, $q(\tau)=\max\{0,\,q_1+s_{lo}[\ln\tau-\ln\tau_1]\}$. This object provides `ppf`, `cdf`, $P(Y>\theta)$, $P(\text{category})$ and exact-integral CRPS.
**Measured:** against analytic CRPS of lognormal ($\sigma\in\{.3,.5,.8\}$) and gamma distributions, the estimator's relative error is **0.2 % mean (≤ 0.5 % max) at $K=19$ but 1.0–3.5 % mean (up to 6.2 % max) at $K=7$**, and 13 % for a naïve 7-level estimate without a tail model. That is why the scoring grid has 19 levels.

**Monotone rearrangement.** Independently trained quantiles can cross (in a synthetic test, 7 of 500 rows). Sorting the $K$ values restores monotonicity and **never increases the total pinball loss**: $f(\tau,q)=\rho_\tau(y-q)$ has $\partial^2f/\partial\tau\,\partial q=-1<0$ (submodular), so by the rearrangement inequality $\sum_kf(\tau_k,q_{\pi(k)})$ is minimised by the sorted pairing (cf. Chernozhukov et al., 2010).

## 6. LightGBM quantile stacker

**Δ — decoupled stacker design.** The plan states that LightGBM "natively supports multi-quantile regression" through an `alpha` array. It does not: `alpha` is a single `double`; passing `[0.1, 0.5, 0.9]` raises *"Parameter alpha should be of type double"* (tested, LightGBM 4.7). SmogSense trains $3\times19=57$ boosters with `objective=quantile`, `alpha=<τ>`, then applies rearrangement.

**Inputs.** $[\,z\ (32)\ \|\ \text{block covariates}\ \|\ \text{fire}\ \|\ \text{recent bias}\ \|\ \text{calendar}\ \|\ \text{static}\ \|\ \text{cams\_lead\_offset\_h}\,]$. The tree model's job is the *abrupt, non-linear interactions* of the plan's example: high upwind fire only matters when the wind aligns **and** the boundary layer is shallow.

**Regularisation for tails.** A leaf should hold at least two expected tail observations: $n_{\rm leaf}\cdot\min(\tau,1-\tau)\ge2\Rightarrow$ `min_data_in_leaf` $\ge40$ at $\tau=0.05$ (the default).

**Stacker construction.** The source domain is split by season blocks into E (encoder meta-training), S (stacker training), V (tuning / early stopping), and T (Delhi confirmatory). There is no cross-fitting: independently trained encoders have incoherent latent frames. Final encoder = E only.
Stacker rows come from **simulated deployment** on S: adapt on a sparse-task support window, freeze, extract latents on the query window, store the row. The bundle binds the `encoder_sha256` to the stacker manifest. Models M6 and M7 each own their own stacker. There is no warm-start residual boosting and no neural/GBM blend in the product path (both are kept as ablations).

## 7. Sparse-network task construction

**Δ — the plan conflates two different "N"s.** In the plan's sensor-sparsity experiment, "zero local stations" is described as pure zero-shot transfer, but the encoder consumes the *target station's PM2.5 history*; with no local station there is no history to consume. The consistent definition: $N$ is the number of **available stations**, which supply *both* the labels used for adaptation *and* the observed-history input at inference.

A **task** $\mathcal T=(d,p,A,B,W^{\rm sup},W^{\rm qry})$: domain $d$; season $p$; available set $A\subset\mathcal S_d$ with $|A|=N\sim\mathrm{Unif}\{0,1,2,3,5,8\}$; held-out set $B=\mathcal S_d\setminus A$ (capped for speed); an earlier support window and a later query window separated by a 72-hour embargo.

* **History input.** For any target $s$: $H_A(s,t)=\sum_{a\in A}w_a(s)\,x_{a,t}\big/\sum_aw_a(s)$ with inverse-distance weights $w_a(s)\propto\mathrm{dist}(s,a)^{-2}$ within 25 km (plus a "distance to nearest available" feature). If $N=0$ the channel is fully masked: **sensor-free mode**, driven by CAMS, meteorology and fire.
* **Two evaluation modes.** *Own-history* (forecast the future at stations in $A$) and *neighbour-history* (forecast at $B$ using $A$ — i.e. at places without a sensor). Training samples draw "own history present" with probability 0.5. The sparsity curve's primary mode is **leave-stations-out** (neighbour-history).
* **Modality and group dropout.** With probability 0.3 the whole PM2.5 history is masked (so sensor-free mode is trained, not merely tested); fire and AOD groups are each dropped with probability 0.1 so a failed FIRMS call degrades gracefully.

Meta-training tasks come from every Delhi station-season (≈ hundreds). Optional extra source domains (Amritsar, Ludhiana) add task diversity if Delhi-only transfer is brittle.

Season-block split of the source domain, e.g. 8 seasons → E: oldest 4, S: next 2, V: 1, T: 1 (by availability; at least one season each). Station identity is `sensor_id`. The N=0 behaviour from ADR-017 item 7.

Season-block split of the source domain, e.g. 8 seasons → E: oldest 4, S: next 2, V: 1, T: 1 (by availability; at least one season each). Station identity is `sensor_id`. At $N=0$, the behaviour is from ADR-017 item 7: too little support or rejection leads to meta-init latents, stacker unchanged, level 0, and adaptation_status recorded.

## 8. Meta-learning (MAML) and adaptation

**Setting.** Neural parameters $\theta=(\theta_{\rm enc},\theta_{\rm head})$ where the head is a *temporary* 3-level × 3-horizon non-crossing head. Task loss $\mathcal L_{\mathcal T}(\theta;D)=\frac1{|D|}\sum_{(x,y)\in D}\ell\big(f_\theta(x),y\big)$ with $\ell$ the masked mean pinball loss over all levels and horizons.

**Inner loop** (support set $D_i^{\rm sup}$, $J$ steps, rate $\alpha$):

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

The plan's "gradient of the gradient" is exactly the Hessian product. **Variants** (config `meta.algorithm`): *FOMAML* sets $H\approx0$; *Reptile* updates $\theta\leftarrow\theta+\epsilon\,\frac1n\sum_i(\theta_i^{(J)}-\theta)$ (Nichol et al., 2018); *ANIL* adapts only the head in the inner loop (Raghu et al., 2020: most MAML benefit is feature reuse); *second-order MAML* is available for $J\le3$.
**Default: first-order, $J=3$**, on CPU via `torch.func.functional_call`. Evidence that second-order buys little at these sizes is part of the ablation, not an assumption.

**Deployment adaptation (Lahore).** At each issuance: start from the *meta-initialisation* $\theta$ (never from yesterday's adapted weights, so adaptation is a deterministic function of data and cannot drift). One shared recipe block `adaptation.inner` (loss, $\alpha$, $J$, support window, masks, clip, adapted parameters) is used by meta-training, S-task simulation, and deployment. The support window is sampled from {7, 14, 30} days in meta-training and deployment uses the same set. Deployment $J$ equals meta-training $J$; any other $J$ is an ablation.

**Acceptance and drift rules.** Acceptance requires chronological fit and a 72-hour embargo. Adaptation is accepted if and only if the check pinball loss improves by $\ge 2$ % and on $\ge 60$ % of check days, AND the latent-drift statistic $D \le$ the 99th percentile observed over simulated S tasks. The drift statistic $D$ is the mean over latent dimensions of $|\mu_{\rm target} - \mu_{\rm ref}|/\sigma_{\rm ref}$, with the threshold calibrated on simulated S tasks. If rejected, the system falls back to the meta-initialisation latents.

## 9. Calibration

**CQR** (Romano et al., 2019). With scores $E_i=\max\{\hat q_{lo}(x_i)-y_i,\ y_i-\hat q_{hi}(x_i)\}$ on a calibration set of size $n$ and $\hat Q$ the $\lceil(1-\alpha)(n+1)\rceil$-th smallest, the interval $[\hat q_{lo}-\hat Q,\ \hat q_{hi}+\hat Q]$ covers with probability $\ge1-\alpha$ **under exchangeability**.
Daily time series are not exchangeable, so the target miscoverage is tracked with **adaptive conformal inference** (Gibbs & Candès, 2021):

$$
\alpha_{t+1}=\alpha_t+\gamma\,(\alpha-\mathrm{err}_t),\quad \mathrm{err}_t=\mathbf 1\{y_t\notin C_t(\alpha_t)\},\qquad
\Big|\tfrac1T\sum_{t\le T}\mathrm{err}_t-\alpha\Big|\le\frac{\max(\alpha_1,1-\alpha_1)+\gamma}{\gamma\,T}\ \ \text{for any sequence}.
$$

A horizon-$h$ label arrives $h$ hours late, so each horizon keeps its own $\alpha$-state and is updated when its label becomes available (delayed-feedback variant; stationary-error guarantee is heuristic under delay and cross-station dependence — hence it is evaluated, not assumed).
Defaults: nominal 80 % interval, $\alpha=0.2$. ACI default $\gamma=0.02$, tuned over $\{0.01, 0.02, 0.05\}$ on dev. The bound values are (1.79, 0.46, 0.19 at $T=90$). ACI is an adaptive correction that is evaluated, not guaranteed. 60-day window, ≥ 200 samples; lower and upper tails adapted separately. Isotonic recalibration of $\tau\mapsto$ empirical frequency (Kuleshov et al., 2018) is the alternative. Calibration is model M8 in the ladder and is accepted only if it improves coverage without worsening CRPS.

## 10. Model ladder and ablations

Complexity must *earn* its place (the plan itself says the AI is unjustified if it cannot beat CAMS). Each rung is scored with identical data, splits and metrics ([evaluation](evaluation-strategy.md)).

| ID | Method | Purpose / what it isolates |
|---|---|---|
| M0 | Persistence, residual quantiles from the available set A only; undefined at N=0 | naïve floor |
| M1 | Raw CAMS; deterministic at N=0, residual-quantile form from A at N≥1 | the physics-based forecast |
| M2 | CAMS-BC from A only; equals M1 at N=0 | the *fair* baseline: any local data at all |
| M3 | Climatology from A only; undefined at N=0 | probabilistic reference without skill |
| M3b | Linear quantile regression on {CAMS PM2.5, ln VC, RH, recent bias, fire E24}, fit on A's support window | simple learned baseline |
| M4 | Tabular LightGBM trained on Lahore A only | "from scratch" (plan: fails at small N) |
| M5 | Tabular LightGBM trained on Delhi, no weight adaptation; features computed from A | pure transfer (feature-based adaptation) |
| M6 | Decoupled pipeline with a supervised-pretrained encoder, same recipe, own stacker | strong simple transfer baseline |
| M7 | Decoupled pipeline with a meta-learned encoder, own stacker (headline candidate) | the proposed system |
| M8 | M7 plus CQR/ACI | calibration layer |

Headline model (HM) = the highest rung meeting the promotion rule on the dev window; ties go to the simpler rung. Add ablation A-warm (warm-start residual boosting) and A-blend.

**Promotion rule.** A component stays if its CRPS improvement over the previous rung has a 95 % block-bootstrap interval excluding 0 at ≥ 2 of 3 horizons and coverage is not worse. **Ablations:** −fire, −AOD, −ventilation features, −recent bias, −static; GRU vs TCN vs no encoder (tabular lags only);
FOMAML vs ANIL vs second-order vs Reptile; Delhi-only vs Delhi+Amritsar+Ludhiana; raw vs log1p target; 7 vs 19 levels; with and without modality dropout. **H5** (pre-registered): M7 beats M6 by ≥ 3 % CRPS at $N\in\{1,2\}$ — if not, that is the finding. The literature gives real reasons to expect simple fine-tuning to be competitive.

## 11. Training protocol and reproducibility

* **Splits.** Source: train on seasons $\le S-2$, validate on $S-1$, test on $S$; spatial hold-out of stations; 72-hour embargo between blocks; purge any sample whose lookback or target window overlaps another split (López de Prado, 2018).
* **Determinism.** `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, fixed thread counts, LightGBM `deterministic=true, force_row_wise=true`; the manifest records library versions, config hash, data partition hashes, CAMS cycles in the training window.
* **Tuning.** Seeded random search (30 trials) on $\tau\in\{.1,.5,.9\}$; no extra dependency, no paid tracker. Results are written to `data/processed/runs/<run_id>/` as Parquet and Markdown.
* **Compute (estimates to be measured in Phase 2 and recorded in the model card).** GRU pre-training: tens of minutes on 2 cores; first-order meta-training ($3{,}000$ outer iterations × 8 tasks × 3 inner steps): on the order of 1–2 h; LightGBM $3\times19$ boosters: ~10 min; the sparsity experiment is sharded by $N$ across matrix jobs (each < 6 h). All of it is CPU-only and runs on free resources.
* **Fallbacks.** If a bundle fails integrity or compatibility, the daily pipeline serves the CAMS-BC baseline (degradation level 2).

## 12. Failure analysis and known risks

| # | Risk | Mitigation / test |
|---|---|---|
| R1 | **Negative transfer.** Delhi is *downwind* of the burn belt; Lahore is *adjacent*, so lag structure and source mix differ | Target-centric features; H4/H5; optional Punjab source cities; always report $N=0$ honestly |
| R2 | **CAMS upgrades shift distributions** (the model cycle changes about yearly; a spring-2026 upgrade was followed by publication delays) | Record cycle; evaluate before/after; retrain trigger |
| R3 | **Right-censoring of sensors** near their effective range (~500 µg m⁻³ typical for optical sensors) biases upper quantiles low | `HIGH_RANGE` flag; verify tails on reference monitors; consider censored pinball |
| R4 | **Train/serve skew** | Same product (CAMS archive) and same as-of function for both; ERA5 never substituted |
| R5 | **Few-shot instability** | Adapt from the fixed meta-initialisation; fall back to $\theta$ if support loss rises |
| R6 | **Representativeness** (40 km grid vs a point) | Static features, recent-bias feature, station-level adaptation |
| R7 | **Network non-stationarity** (sensors appear/disappear) | Sparse-network training; $N$ logged per forecast |
| R8 | **Public over-confidence in intervals** | Natural-frequency wording; "8 days in 10 / 1 day in 10"; coverage monitoring |
| R9 | **Leakage** via scaling, tuning or overlapping windows | Source-only scalers; embargo + purging; unit tests on as-of joins |
| R10 | **Emission regime change** (policy, kiln technology) | Rolling coverage monitor; retrain trigger |
| R11 | provisional-vs-settled truth skew |  |
| R12 | latent drift on real low-cost support |  |
| R13 | adaptation effect smaller than assumed |  |
| R14 | RH-mask tail truncation |  |
| R11 | **Provisional-vs-settled truth skew** | Track reporting differences |
| R12 | **Latent drift on real low-cost support** | Drift statistic guard |
| R13 | **Adaptation effect smaller than assumed** | Evaluate robustly |
| R14 | **RH-mask tail truncation** | Mask ablation |

## References

Alduchov & Eskridge (1996) *J. Appl. Meteorol.* — Magnus coefficients. · Bai, Kolter & Koltun (2018) — TCN. · Barkjohn, Gantt & Clements (2021) *Atmos. Meas. Tech.* — PurpleAir correction. · Che et al. (2018) *Sci. Rep.* — GRU-D.
· Chernozhukov, Fernández-Val & Galichon (2010) *Econometrica* — quantile rearrangement. · Cho et al. (2014) — GRU. · Diebold & Mariano (1995) *JBES*. · Finn, Abbeel & Levine (2017) *ICML* — MAML. · Gibbs & Candès (2021) *NeurIPS* — ACI.
· Gneiting & Raftery (2007) *JASA* — proper scoring rules. · Gneiting & Ranjan (2011) *JBES* — threshold-weighted CRPS. · Harvey, Leybourne & Newbold (1997) *Int. J. Forecast.* · Hersbach (2000) *Weather Forecast.* — CRPS. · Ke et al. (2017) *NeurIPS* — LightGBM.
· Koenker & Bassett (1978) *Econometrica* — quantile regression. · Kuleshov, Fenner & Ermon (2018) *ICML*. · López de Prado (2018) *Advances in Financial ML* — purging/embargo. · Matheson & Winkler (1976) *Manage. Sci.* — CRPS.
· Nichol, Achiam & Schulman (2018) — Reptile. · Politis & Romano (1994) *JASA* — stationary bootstrap. · Raghu et al. (2020) *ICLR* — ANIL. · Romano, Patterson & Candès (2019) *NeurIPS* — CQR.
· arXiv:2108.00640 — few-shot calibration of low-cost PM2.5 sensors (cited in the original plan).
