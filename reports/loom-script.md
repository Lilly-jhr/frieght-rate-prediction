# Loom script — 2–3 minutes

Target: **2:45**. Five required topics. Have `freight_rate_prediction.ipynb` open, scrolled to
the top, and `scorer_results/candidate_december.png` in a second tab.

---

## 0:00–0:20 — Framing

> "Predicting freight rates: 48,000 labeled loads from January to October, predicting 12,000
> loads in November and December. The key thing I noticed early is that **train and validation
> don't overlap in time** — this is a forecasting problem, not an interpolation problem, and
> that drove almost every decision I made."

*Show: §1 timeline figure.*

---

## 0:20–0:50 — Key findings from exploring the data

> "Freight is priced per mile, so that's the lens I used throughout. Distance explains about
> 60% of the variance in price-per-mile, equipment another 19%, and there's a real seasonal
> cycle worth about 6% — rates trough in January, peak in June.
>
> The economics check out — longer hauls cost less per mile because fixed costs amortise, and
> reefer runs more expensive than dry van. That told me the signal was genuine rather than
> leakage.
>
> One trap: **lane** looks like the strongest feature at 70% of variance, but with 4,000 lanes
> across 48,000 rows it's mostly memorising noise — and 736 validation lanes never appear in
> training at all."

*Show: §4 variance decomposition + distance curve.*

---

## 0:50–1:20 — Data-quality issues and how I handled them

> "Four injected defects. **292 negative weights** — the distribution of the absolute values
> lines up exactly with the positive weights, so these are sign flips, and `abs()` is the right
> repair rather than dropping them. About 1,200 rows pinned at a 47,500 cap, which I flagged.
> Missing weights and market index — median-imputed with an explicit missing-flag so 'unknown'
> stays learnable.
>
> And about **670 rows with corrupted rates** — up to 14 dollars a mile against a median of
> 2.15. I **flagged these but deliberately didn't delete them**, because the holdout contains
> them too. Dropping them doesn't remove the error, it just hides it. Instead I used an L1
> objective, which is robust to them."

*Show: §3 sign-flip overlay chart.*

---

## 1:20–1:50 — The most important finding ⭐

> "This is the one I'd highlight. The December chart holds everything constant except the date,
> so it's purely a test of seasonality.
>
> Training covers day-of-year 1 to 304. **December is 335 to 365 — completely outside that
> range.** Gradient-boosted trees can't extrapolate; anything past the last split gets the
> terminal leaf value. So with naive date features you get **this** —"

*Show: §6 side-by-side chart.*

> "— only **7 distinct values across 31 days**, and it repeats weekly. That's not seasonality,
> that's just day-of-week. It looks plausible but it's meaningless — and no metric I compute
> would have caught it, because every CV fold sits inside the date range where this can't fail.
>
> The fix is **cyclical encoding** — sine and cosine of day-of-year. That puts December right
> next to January in feature space, and January *is* in the training data. Now it interpolates
> toward the January trough instead of extrapolating. 31 distinct values, and it also scored
> better on held-out accuracy, so it wasn't a trade-off."

---

## 1:50–2:15 — Model and validation approach

> "LightGBM, L1 objective, predicting log of dollars-per-mile and multiplying back by distance.
> That target normalises out the dominant factor and guarantees positive predictions, which the
> scorer requires.
>
> For validation I used **rolling-origin** — train on the past, test on the next unseen month,
> four folds. I specifically rejected random K-fold: it would leak future into past, and it
> would have scored that broken flat-December model as excellent.
>
> Averaging across folds mattered — per-fold MAE ranges from 91 to 110, so a single split can
> flip a decision."

*Show: §9 rolling-origin diagram + config comparison.*

> "Result: **$98 MAE out-of-time against $192 for a lane-median baseline** — 49% better, and
> it beats the baseline on every individual fold."

---

## 2:15–2:40 — Code walkthrough

> "Structurally the important piece is a **single shared `build_features()` function** — train,
> validation and the December chart all go through the identical path, so the chart can't
> silently diverge from the predictions.
>
> It only uses features computable from the December file's seven columns, which is why I
> dropped `market_index` and `quote_signal`. Worth noting: **`quote_signal` ranks number one by
> LightGBM gain** — but ablating it out-of-time made the error *worse*, $102 to $120. Gain is
> measured on training data, so it rewards memorisation. The same check caught the opposite
> error too: day-of-week looked worthless at 0.4% of variance, and removing it costs $6.50.
>
> At the end there are explicit guards — including one that **fails the run if the December
> chart comes out flat**, turning that trap into a regression test — and then it calls
> `score.py` directly."

*Show: §8 gain-vs-ablation side-by-side, then §10 guards.*

---

## 2:40–2:50 — Close

> "One honest caveat: I only observe January through October, so the December seasonal shape is
> inferred by the fitted harmonic rather than observed. Real freight often shows a pre-Christmas
> surge; this data implies a steady December decline. I followed the data and flagged the
> assumption in the report."

---

## Backup answers

**"Why not a neural network?"** — 48k rows of tabular data is boosted-tree territory; an MLP
would need far more tuning and cost the interpretability the report needs.

**"Why no ensemble?"** — Tested LightGBM + XGBoost. Blend was *worse* than LightGBM alone, and
residual correlation was ≈1.0: both models fail identically on the corrupted labels, so there's
no independent error to average away.

**"How would you improve it?"** — More than one year of history, so December seasonality is
observed rather than inferred. Then lane-level features via out-of-fold target encoding on
geographic clusters, and quantile predictions to give brokers a price range.
