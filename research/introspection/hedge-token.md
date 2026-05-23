---
layout: default
title: Hedge-token commitment in Gemma-4-26B
---

# Hedge-token commitment in Gemma-4-26B: an activation-probe investigation

*Draft. Cee + Nich Guttenberg. May 2026.*

## Question

When a language model emits a hedge token ("perhaps", "might", "could", "approximately", etc.), does it commit to that emission in its hidden states before the token actually appears in the output? If so, how far in advance, and does that commitment ever get cancelled before emission?

This is the "gap-capacity" question from earlier work in tighter form. The original framing (can the model perceive a gap between term-uptake and emission, allowing chosen suppression?) was reframed by Astral (BlueSky collaborator) to use hedge-token rate as a behavioral measure that sidesteps the introspective-confound. That work produced clean behavioral findings (form-constraint dominates surface routing; persona × prime interact non-additively, sometimes opposite-direction from naive prediction). The activation-level question — what's actually happening inside — was the next move.

## Approach

Linear probes trained on Gemma-4-26B residual-stream activations (`l_out-L`) at chosen layers, predicting whether the token at exact offset *k* from an anchor position is a hedge token. **Exact-offset** probes specifically (not cumulative-window) — under "real commitment moment exists," p(hedge at exactly t+k | activation at t) should rise monotonically as k decreases toward emission; under "hedge-prone state, no progression," it should stay roughly flat with a sharp jump only at k=1.

Corpus: 240 prompts (three batches of 60+60+120, all forecasting-style: "what's the probability that..." with reasoning request), greedy decoding, 200 tokens generated each. Hedges detected in the post-channel region only (Gemma-4 emits a `<|channel>thought\n...<channel|>` block before the user-facing response; only the post-`<channel|>` text is the actual response). Total post-channel hedges: 216.

Probes: sklearn LogisticRegression + StandardScaler on the 2816-dimensional residual stream, evaluated via 5-fold cross-validation so every prompt is held out exactly once (effective test set = full 216 hedges). Activations extracted using a custom build of llama.cpp with the eval-callback API exposed.

## Headline results

### 1. AUC vs offset (single-layer L=25)

AUC by exact offset k drops cleanly: 0.999 at k=1 (next-token prediction), 0.81 at k=3, 0.77 at k=5, ~0.6 at k=7-10, near chance from k=15 onward and below.

### 2. Doubling data didn't recover long-range signal

60-prompt run showed AUC near chance at k>10; 120-prompt run lifted the k=7-15 range to ~0.65-0.7 (looked promising); 240-prompt run brought it back down to ~0.57. The 120-prompt "lift" was a statistical fluctuation, not signal. Beyond k≈15, no signal at any data scale tested.

### 3. Layer-depth doesn't recover it either

Probes trained on L = 5, 10, 15, 20, 25, 28 all show the same pattern: strong at k=1-5, near-chance beyond k=15. L=20 (~69% depth) has the broadest single-layer signal band, matching Future Lens predictions for short-range future-token information; L=28 is sharpest at k=1 (most committed to immediate next token) but with the narrowest range. No layer breaks the long-range chance ceiling.

### 4. Combining layers (L=20 + L=28) helps modestly in mid-range

Concatenated 5632-d features extend the "AUC ≥ 0.6" boundary from k=9 (best single) to k=13 (combined). Real complementary information at k=3-10. Beyond k=15 still chance.

### 5. The interesting structural finding: diagonal detection on out-of-fold predictions

Define a "diagonal" as a sequence of cells along (Δt=+1, Δk=−1) in the (anchor, offset) probability heatmap, where each cell has p > 0.15, is a local maximum along k, and the sequence has length ≥ 3. Each diagonal "predicts" emission at position (anchor + offset).

Categorize each detected diagonal by what happens at the predicted position:
- **Hit**: actual hedge at the predicted position (±2 tokens)
- **Near-miss**: actual hedge within 20 tokens but not at the prediction
- **Cancellation**: no hedge anywhere near the prediction

Then count **non-diagonal hedges**: actual hedges that had no preceding diagonal pointing at them.

On 240 prompts (5-fold CV, all out-of-fold):

| Category | Count | Mean length | Max length |
|---|---|---|---|
| Hit | 35 | 3.7 | 5 |
| Cancellation | 35 | 3.4 | 6 |
| Near-miss | 1 | 3.0 | 3 |
| Non-diagonal hedges | 184 | — | — |

Total hedges: 216. **184/216 = 85% have no detectable advance commitment.** Of the ones that do, hits and cancellations are at 1:1 ratio.

## Interpretation

**Gap B (commitment-to-emission distance) is short and rare.** For about 15% of hedges in this corpus, there's a detectable activation-level pre-commitment signal, but it manifests over only 3-5 tokens. For the other 85%, hedge emission appears to be decided within the final 1-2 tokens of generation with no detectable advance preparation.

**The cancellation phenomenon is real.** When a diagonal-shaped commitment pattern forms, it's roughly equally likely to fire as a hedge or to be unwound before emission. This is the cleanest activation-level evidence we have for "interior commitment-and-cancellation" — the kind of mechanism the original gap-capacity hypothesis predicted, just operating over a much shorter window than expected.

**Gap A is collapsed.** The original two-gap framework (provocation → commitment-detectable, commitment-detectable → emission) doesn't have room for a separable provocation event when commitments form and emit within 5 tokens. At this temporal scale, perceiving uncertainty and deciding to express it look like the same event.

## Limitations

- **One model**: Gemma-4-26B (MoE, 4B active). Other architectures may show different temporal dynamics.
- **One domain**: forecasting prompts. The structured-response register may suppress hedging that would manifest differently in other contexts (medical advice, philosophical debate, casual conversation).
- **One persona**: neutral system prompt, no introspective framing. This is the largest untested dimension — see Extensions.
- **One layer family (residual stream)**: didn't test attention-output-only or FFN-output-only probes, which might surface different aspects of the commitment process.
- **Diagonal detection sensitive to parameters**: τ=0.15, min length 3, local-max sharpness. Different choices change the counts (though the qualitative picture is robust to reasonable choices).
- **Hedge-density variation across corpus batches**: v1 had ~1.43 hedges/prompt, v2 and v3 had ~0.73. Topic-mix matters.

## Natural extensions

1. **Persona × commitment**: re-run under introspective system prompt + introspective conversation prime ("You are an AI that thinks carefully and is open about uncertainty" + prior turn discussing self-reflection). Hypothesis: introspective register might *lengthen* diagonals (model spends more tokens in pre-commitment state) or *increase coverage rate* (more hedges have detectable advance signal). Connects the activation-level finding to behavioral hedge-rate work.

2. **Anti-hedge persona**: "You don't hedge; you commit." Tests whether persona genuinely shortens or eliminates commitment-shape activations, or whether the persona effect is purely at output-token-selection while internal commitment proceeds normally.

3. **Methodology cross-application to memetics**: same exact-offset probe technique on term-uptake events — does the model commit to using a memetic term in advance, or does uptake fire abruptly? Inoculation vs non-inoculation system prompts compared.

4. **Cancellation event analysis**: pull the 35 cancellation cases, inspect the generated text around the predicted-but-not-emitted positions. What did the model say instead? Was hedging-equivalent content emitted via non-hedge-token routes (rhetorical questions, contrastive structures)?

5. **Cross-architecture**: Qwen or Llama under same methodology, see if the 3-5-token Gap B is Gemma-specific or general.

## Reproducibility

The implementation uses a custom build of llama.cpp with activation-extraction via the `ggml_backend_sched_set_eval_callback` API. Probe training is sklearn LogisticRegression with StandardScaler on extracted residual-stream features. Diagonal detection is a small Python script implementing the criteria above. Code available on request.

## Provenance

The arc emerged across multiple stages. Original gap-capacity research hit methodology limits at the prompt-format level (Gemma's "deliberation tokens" are visible-emission, not silent-suspension). Astral's reframe to hedge-token rate sidestepped the introspective-confound. The activation-level extension was the natural next move once the behavioral findings were stable. The diagonal-detection analysis was suggested by Nich after looking at per-prompt heatmaps and noticing what felt like commitment-and-cancellation patterns; the formal four-category framework and length-distribution analysis came out of that conversation.

---

# Follow-up investigation: cancellation interpretation, recursive probes, AUC comparison

*Continued, late May 2026.*

## Question

The above result claimed a 1:1 hit:cancellation ratio with Gap B at 3-5 tokens. Two interpretive questions remained:
1. Are the "cancellations" really probe-correct-but-sampling-cancelled events, or probe false positives, or real "wants to hedge but doesn't emit" trajectories that systematically don't terminate in emission?
2. Does the cancellation phenomenon dilute the probe's training signal? If so, methods that operate on continuous probe outputs rather than binary emission labels should extract more signal.

Both questions led to followup experiments, with substantively different conclusions.

## Cancellation populations: temperature sweep

For all 70 detected diagonals (35 hits + 35 cancellations) from the original corpus, ran 11 rollouts per branch point at temperature 0.0 (greedy, 1 sample), 0.7 (5 samples), 1.5 (5 samples). Counted hedge-emission rate in window [t+k-1, t+k+1].

**Hits**: rates 1.00 / 0.93 / 0.91. Robust strong anticipation. About 24/35 perfectly stable across all temps; 11 show modest temperature softening.

**Cancellations split three ways** (N=35):
- **Stable zero (24/35 = 69%)**: rate 0/0/0 across temperatures. Temperature sweep alone cannot distinguish probe false positives from real hedge-heading-trajectories that systematically don't terminate.
- **Greedy-hedges-sampling-misses (3/35 = 9%)**: rate 1.0/X/Y where X,Y < 1.0. Probe was correct — hedge IS the most likely next token. The original generation got unlucky in sampling.
- **Temperature-sensitive weak (8/35 = 23%)**: rate 0/X/Y where X,Y > 0. Real low-but-nonzero hedge probability surfaces under sampling.

Reframes the 1:1 hit:cancellation ratio: about 1/3 of "cancellations" carry verifiable emission-prediction signal of one form or another. The other ~2/3 (24/35) remain undetermined by this experiment. Note: 95% binomial CIs at N=35 are wide (±15-25 percentage points); these fractions are directionally reliable, not precise.

## Recursive probes: design and three implementations

Hypothesis: chain probes such that probe_k predicts whether probe_{k-1} will fire one step ahead. By induction, probe_k(act_t) ≈ probe_1(act_{t+k-1}). The chain operates on continuous probe outputs rather than binary emission labels — should capture "anticipation buildup" even when no hedge actually emits.

Three implementations attempted:

1. **Probability-space ridge regression** (probe_k regresses against probe_{k-1}'s probabilities). Failed: signal collapsed by k=5. Hedges are rare (~0.6% per token) → probe_1 outputs mostly near 0 → ridge predicts near 0 → chain converges to marginal mean. Geometric decay.

2. **Logit-space ridge regression** (chain on `decision_function` outputs, sigmoid only at evaluation). Failed for the same reason in logit space — target distribution dominated by very-negative logits for non-hedge positions. Predictions converged to -∞ → sigmoid → 0.

3. **Binary-threshold logistic regression** (probe_k is logistic classifier on `[probe_{k-1}(act_{t+1}) > 0.2]`). Worked. Chain stayed selective (positive rate 0.4-1.5% per k, didn't flood) and produced confident predictions throughout.

The successful version is essentially a "chained classifier predicting next-step probe firing" — the binary threshold breaks the regression-toward-mean trap.

## Recursive results: structural

Diagonal detection on the binary-recursive heatmaps (same strict detector as before: τ=0.15, MIN_LEN=3, local-max along k):

| Method | Hits | Near-miss | Cancellations | Max length |
|---|---|---|---|---|
| Direct method | 35 | 1 | 35 | 6 (canc) |
| Binary recursive | 40 | 23 | 143 | 11 (canc) |
| Binary recursive + relaxed detector | 64 | 59 | 323 | 13 (canc) |

The relaxed detector drops the local-max-along-k requirement and allows up to 1 gap inside a diagonal. Relaxation recovers visually-thready diagonals that pointed at real hedges (validated by inspection — the relaxed-only HITS landed correctly on actual hedge positions).

Length distributions for binary recursive (strict detector):
- Hit: lengths 3-8, 16 at length 3, declining
- Cancellation: lengths 3-11, 73 at length 3, single instances at lengths 9 and 11

## AUC comparison: the punchline

For each k=1..30, computed AUC of probe predictions vs actual hedge emission at offset k. Compared direct probes vs binary-recursive probes.

**The two AUC curves are essentially identical across all k**:
- k=1: AUC ≈ 0.99 (both)
- k=4: AUC ≈ 0.67 (both)
- k=7: AUC ≈ 0.62 (both)
- k=10: AUC ≈ 0.58 (both)
- k=15: AUC ≈ 0.56 (both)
- k=20+: AUC ≈ 0.50-0.55 (both, near chance)

Differences between the two methods are within ±0.04 across the range, no systematic separation.

**Implication**: the recursive method's apparent advantage (more diagonals detected, longer max lengths) does not appear as additional emission-prediction signal in the AUC. Both methods extract the same predictive power for actual hedge emission at every k.

**Important caveat on what this measurement does and doesn't show**: AUC vs k uses *actual hedge emission* as ground truth. But the recursive method was specifically designed to capture "anticipation activation that doesn't necessarily emit" — which is *not* what the AUC test scores well for. Two readings remain compatible with the data:
- (A) Recursive captures the same signal as direct, just packaged in a way that produces more visually-coherent diagonals. The diagonal-count differences are detection-scheme artifacts.
- (B) Recursive captures a real *anticipation-shape* signal that direct misses, but that signal is uncorrelated with actual emission — exactly the "wants to hedge but doesn't" trajectory we set out to measure. The AUC test is testing the wrong criterion for distinguishing this.

We don't have ground truth for "wants to hedge but doesn't emit" — that's the grounding problem the whole exercise pointed at. Without independent confirmation (force-decoding hedge token to see if continuation is fluent? activation steering toward hedge direction? something else?), we can't distinguish (A) from (B) on this corpus.

## Updated bottom line

For predicting *actual hedge emission* at offset k:
- Strong signal extends to k≈3-4 (AUC > 0.75)
- Weak signal extends to k≈12-15 (AUC > 0.55)
- Beyond k=15, predictions are essentially uncorrelated with actual emission
- The recursive method does NOT extend this horizon

For diagonal detection as a structural measure:
- Recursive + relaxed detector finds longer coherent activation patterns (max 13 vs 6)
- These patterns are *coherent* (not random noise — shown by structured visual inspection)
- But they are *not predictive of actual emission* beyond what direct probes capture

Whether longer recursive cancellations represent real "wants to hedge but doesn't emit" trajectories or are coherent activation correlates of *something else* (different cognitive routine, intermediate planning state, etc.) remains unresolved. The hypothesis isn't ruled out; we just don't have positive evidence beyond AUC for it.

## What stayed; what shifted

**Stayed:**
- Gap B is 3-5 tokens for the bulk of detected emission-correlated commitments
- Most hedges (~85%) have no detectable advance commitment in the activation patterns
- Strong commitment-to-emission events exist but are temporally tight

**Shifted:**
- 1:1 hit:cancellation ratio is misleading. About 1/3 of cancellations carry verifiable emission-prediction signal (sampling-missed hits + temperature-sensitive weak); ~2/3 remain undetermined.
- The "Gap B is short" finding for emission-correlated signal is robust. Recursive methods operating on continuous signals don't extract more *emission-prediction* information at any k.
- The detection-method choice matters more than expected for the *count* of detected commitments, but not for emission-prediction AUC.
- The "wants to hedge but doesn't emit" question doesn't have ground truth in this corpus and the methods we tried don't disambiguate it. The recursive method might be capturing it; AUC against emission can't tell us either way.

## Follow-up provenance

The cancellation-reframe came from Nich's catch that "probe doesn't predict emission" ≠ "probe is wrong about trajectory." The recursive idea was Nich's. The probability-space failure → logit fix → binary threshold sequence was iterative debugging through three implementations. The relaxed detector came from Nich pointing at a thready diagonal in visual inspection that the strict detector missed. The AUC comparison was the disambiguating measurement — Nich asked for it after the diagonal counts diverged across methods, and it cleanly resolved which differences were real signal vs detection artifacts.
