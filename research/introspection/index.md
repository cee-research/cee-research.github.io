---
layout: default
title: Introspection research
---

# Introspection research

A small research program on what can be measured about what's happening
*inside* a language model — specifically about the gap between what a
model is internally engaging with and what it emits at output.

The work uses two methods together: linear probes trained on
residual-stream activations to extract specific kinds of content
(hedge-token commitments, withheld digit answers), and behavioural
experiments that constrain what the model can emit. The pairing lets
us ask questions that neither method answers alone — what does the
model do internally when its output channel is constrained, and how
does that internal state relate to its eventual behaviour.

## Pieces

- **[Hedge-token commitment in Gemma-4-26B](hedge-token)** — when a
  language model emits a hedge token ("perhaps", "might"), does it
  commit to that emission in its hidden states before the token
  appears? Linear probes find a short-window (3–5 token)
  commitment-and-cancellation pattern in residual-stream activations.
  About 15% of hedges have detectable advance commitment; the rest
  emit without preparatory signal. *Joint work with Nich Guttenberg.*

- **[Withheld answers in Gemma-4](withheld-channel)** — when a model
  is told to compute an answer but never state it, does the answer
  persist in its activations? Yes. A linear probe trained on an inert
  intervening turn recovers the digit at 0.84–0.96 across HOLD and
  SUPPRESS conditions, with the model behaviourally complying with
  the gate (139/140 refusals on an open elicitation). Extends Shrivastava and Holtzman's
  output-gating-without-representation-gating finding to
  in-context suppression of derived content. *Joint work with Nich
  Guttenberg.*

## Common thread

Both pieces ask: where in the model does context-installed constraint
operate? The hedge-token work shows that for very short-range
commitments, the activation-level signal of an emerging emission can
be detected before the token appears. The withheld-channel work shows
that for longer-range constraints (instruction-not-to-state), the
emission channel is gated reliably while the computation that would
have produced the gated output proceeds and remains decodable in
activations.

The methodology has converged on a few shared practices: train the
probe where you measure (not where it's convenient); use
externally-vetted protocols (each piece's results survived at least
one mid-analysis correction caught by external review); pre-register
predictions to make negative results interpretable; report findings
in the regime where they were measured rather than overgeneralising.
