"""
views/methodology.py

PAOM Dashboard -- Methodology page. A deeper, centralized reference
for HOW the dashboard's numbers are actually computed, complementing
(not duplicating) the page-specific expanders that explain what a
given number MEANS in context. Scoped to methodology only, by
explicit design -- known limitations and validation caveats live
elsewhere (each recommendation's own confidence tier and disclosure
flags), not on this page.

Every technical result (equation, statistic, correlation) is paired
with a plain-language "what this actually means" explanation in real
baseball terms -- the goal is that a reader with no statistics
background can still understand what's actually been shown, not just
see a correct but opaque number.
"""

import streamlit as st


st.title("Methodology")
st.caption(
    "How PAOM Score and the recommendation engine are actually "
    "computed -- the full, real methodology behind this dashboard's "
    "numbers. For what a specific number means in context, see that "
    "number's own page -- this page is the deeper reference."
)

tab_paom, tab_engine, tab_confidence, tab_validation = st.tabs(
    ["PAOM Score", "Recommendation Engine", "Confidence & Evidence", "Validation & Trust"]
)


# ============================================================
# PAOM SCORE
# ============================================================

with tab_paom:
    st.markdown(
        """
## The Four Components

PAOM Score combines four independently-computed components, each
already scaled to a 0-100 percentile before being combined:

| Component | What it measures |
|---|---|
| **Effectiveness** | Real pitch-quality, modeled against outcomes |
| **Movement** | How much distinct movement-shape territory the whole arsenal covers |
| **Command** | How consistently a pitch is located where aimed |
| **Velocity** | Velocity level and range across the arsenal |

### Effectiveness

Effectiveness is built from a Ridge regression predicting real
`xwOBA` from four real, pitch-level rate stats:
"""
    )
    st.latex(r"\hat{xwOBA} = \beta_0 + \beta_1 \cdot csw\_rate + \beta_2 \cdot chase\_rate + \beta_3 \cdot hard\_hit\_rate + \beta_4 \cdot gb\_rate")
    st.info(
        "**In plain terms:** this is asking, \"based on how often a "
        "pitch generates a swing-and-miss or called strike, how "
        "often hitters chase it out of the zone, how hard it gets "
        "hit when contact is made, and how often it produces a "
        "ground ball, how good is the actual outcome?\" A pitch's "
        "raw Effectiveness score is just this prediction flipped "
        "(lower predicted damage = higher score), then converted to "
        "a 0-100 scale where 50 is league-average and 100 is the "
        "best in the league."
    )
    st.markdown(
        """
### Confidence-weighted shrinkage

Every component shrinks its raw score toward the league average,
based on how much real data backs it:
"""
    )
    st.latex(r"trusted\_score = raw\_score \cdot \frac{confidence}{100} + league\_avg \cdot \left(1 - \frac{confidence}{100}\right)")
    st.latex(r"confidence = \min\left(\frac{n}{saturation}, 1\right) \times 100")
    st.info(
        "**In plain terms:** a pitcher who's only thrown a slider 40 "
        "times this year hasn't shown enough for us to fully trust "
        "whatever number that small sample produces -- it could "
        "easily be a hot or cold streak, not their real talent level. "
        "So instead of showing that raw, possibly-misleading number, "
        "we blend it toward league-average, more heavily the fewer "
        "pitches we've seen. Once a pitcher has thrown a pitch enough "
        "times (the \"saturation\" point), we trust their own real "
        "number fully and stop blending at all."
    )
    st.markdown(
        """
### Combining the four components

The four component percentiles are combined into a single PAOM
Score via a weighted average:
"""
    )
    st.latex(r"PAOM = w_{eff} \cdot Effectiveness + w_{mov} \cdot Movement + w_{cmd} \cdot Command + w_{vel} \cdot Velocity")
    st.info(
        "**In plain terms:** PAOM Score isn't just the four numbers "
        "averaged evenly -- some components matter more than others "
        "for actually predicting real results, so those get more "
        "weight. Effectiveness typically counts for roughly two-"
        "thirds of the total score, since how good each individual "
        "pitch actually performs turns out to matter more than the "
        "other three factors combined. The weights themselves come "
        "from checking, across the whole league, which components "
        "actually predicted real season performance best."
    )
    st.markdown(
        """
### Overall confidence

`paom_confidence` is the **minimum** of the four components' own
confidence values, not an average -- the overall score is only as
trustworthy as its single weakest input:
"""
    )
    st.latex(r"paom\_confidence = \min(conf_{eff}, conf_{mov}, conf_{cmd}, conf_{vel})")
    st.info(
        "**In plain terms:** if a pitcher has plenty of data on three "
        "components but only just started throwing their sinker (thin "
        "data on Movement), the overall PAOM Score's confidence gets "
        "pulled down to match that one weak spot -- not averaged up "
        "by the three strong components. A chain is only as strong as "
        "its weakest link."
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

with tab_engine:
    st.markdown(
        """
## Core Idea

For a candidate change (e.g. "add a slider," "drop the sinker"),
the engine finds real historical pitchers who made that exact kind
of change, weights them by how similar they are to the target
pitcher, and predicts the outcome from what actually happened to
those real comps -- not a fixed rule applied to everyone alike.
        """
    )
    st.info(
        "**In plain terms:** instead of a rule like \"if a pitch's "
        "quality is below X, recommend dropping it,\" the engine "
        "asks a much more direct question: \"has a pitcher who "
        "looks like this one ever actually made this same change "
        "before, and what really happened to them afterward?\" The "
        "recommendation is built from real, similar pitchers' real "
        "outcomes, not a formula applied identically to everyone."
    )
    st.markdown(
        """
## Similarity

Two pitchers' distance is a weighted sum of squared differences
across several real, standardized features:
"""
    )
    st.latex(r"distance = \sqrt{\sum_i w_i \cdot (target_i - comp_i)^2}")
    st.markdown(
        """
The feature set includes seven standardized, tiered features
(arsenal shape at weight 1.0 each -- hull area, nearest-neighbor
distance, fastball-relative break, velocity range, max velocity;
release mechanics at weight 0.5 each -- release position
horizontal/vertical), plus three additional weighted terms: pitch
composition overlap (Jaccard similarity of qualifying pitch types,
weight 1.5), shared-pitch shape distance for pitch types both
throw (weight 1.5), and, for non-ADD candidates specifically,
usage-level matching (weight 2.0). That distance becomes a
similarity weight used directly in the prediction regression:
"""
    )
    st.latex(r"weight = \frac{1}{1 + distance}")
    st.info(
        "**In plain terms:** \"similar pitcher\" doesn't just mean "
        "\"throws hard\" or \"is left-handed\" -- it's a real, "
        "multi-part comparison across how their whole arsenal is "
        "shaped, where they release the ball, whether they throw "
        "the same kinds of pitches, and (for some questions) how "
        "much they already use the pitch in question. Two pitchers "
        "who are close on all of these count as more similar, and "
        "get more say in the final prediction than a pitcher who's "
        "only similar on one or two of them."
    )
    st.markdown(
        """
## The Prediction Regression

For a given candidate, a weighted linear regression is fit across
real historical comps, using their real before/after outcomes:
"""
    )
    st.latex(r"\hat{y}_{after} = \beta_0 + \beta_1 \cdot y_{before} + \beta_2 \cdot usage\_pct\_delta + \beta_3 \cdot existing\_pitch\_quality")
    st.info(
        "**In plain terms:** to predict what happens after this "
        "change, the model looks at three real things: how good the "
        "pitcher already was (a pitcher who was already great tends "
        "to stay closer to great -- this is why the model doesn't "
        "just predict every change as a huge swing), how big a "
        "usage change is actually being made, and how good the "
        "pitch itself is. `existing_pitch_quality` is a real, direct "
        "number for a pitch the pitcher already throws, or a "
        "separate, similarity-based estimate for a brand-new pitch "
        "they haven't thrown yet."
    )
    st.markdown(
        """
## Swap Predictions

A SWAP (dropping one pitch while adding another, as a single
combined decision) is predicted directly from real historical swap
events -- pitchers who made that same kind of combined change --
rather than summed from two separate single-change predictions:
"""
    )
    st.latex(r"\hat{y}_{after} = \beta_0 + \beta_1 \cdot y_{before} + \beta_2 \cdot dropped\_usage\_pct\_before")
    st.info(
        "**In plain terms:** if a pitcher replaces their four-seam "
        "fastball with a sinker, that's really one decision with one "
        "real outcome -- not two separate events that happen to "
        "occur at the same time. This looks at real pitchers who "
        "made that same kind of swap and predicts from what actually "
        "happened to them as a whole, rather than adding together a "
        "separate \"what if they only dropped it\" guess and a "
        "separate \"what if they only added it\" guess, which turned "
        "out to badly overstate the real effect when directly "
        "checked."
    )
    st.markdown(
        """
## Guardrails

- **Anchor-pitch exclusion**: a pitcher's own primary fastball-type
  pitch (their real `anchor_type`) can never be recommended for a
  full DROP.
- **Minimum resulting arsenal size**: no candidate can leave a
  pitcher with fewer than 2 real (≥5% usage) pitch types.
- **High-usage-drop disclosure**: a DROP candidate whose current
  usage exceeds the real, historical 90th percentile for that
  pitch type is flagged, not excluded.
        """
    )
    st.info(
        "**In plain terms:** a pitcher's primary fastball usually "
        "does real, important work the model can't directly measure "
        "-- setting up other pitches, establishing timing -- so it's "
        "never recommended for a full drop, regardless of what the "
        "raw numbers say about that pitch in isolation. Similarly, "
        "the model won't recommend leaving a pitcher with only one "
        "real pitch to throw, and it'll tell you directly when a "
        "drop candidate's usage is unusually high for a real drop, "
        "so you can weigh that yourself."
    )


# ============================================================
# CONFIDENCE & EVIDENCE
# ============================================================

with tab_confidence:
    st.markdown(
        """
## Confidence Tiers

Every recommendation is assigned a confidence tier based on how
many real historical events its own prediction is built from:

| Tier | Real historical events |
|---|---|
| 🟢 Strong evidence | 110+ |
| 🟡 Moderate evidence | 50-110 |
| ⚪ Limited evidence | Fewer than 50 |
        """
    )
    st.info(
        "**In plain terms:** if a recommendation is backed by 150 "
        "real historical pitchers who made a similar change, that's "
        "a much stronger basis for trust than a recommendation "
        "backed by only 20 -- even if both predictions point in the "
        "same direction. The tier tells you at a glance how much "
        "real precedent actually exists behind a specific "
        "suggestion, not how good the suggestion itself is."
    )
    st.markdown(
        """
## PAOM Confidence

As shown on the PAOM Score tab, overall confidence is the minimum
across all four components:
"""
    )
    st.latex(r"paom\_confidence = \min(conf_{eff}, conf_{mov}, conf_{cmd}, conf_{vel})")
    st.markdown(
        """
## Component Confidence

Each individual component's confidence is based on its own real
sample size relative to a saturation point specific to that
component:
"""
    )
    st.latex(r"confidence = \min\left(\frac{n}{saturation}, 1\right) \times 100")
    st.info(
        "**In plain terms:** a low confidence number isn't a "
        "criticism of the pitcher -- it just means there isn't "
        "enough real, established data yet to fully trust their own "
        "number, so the score leans more on the league-average until "
        "more real innings accumulate. It's a reliability label, not "
        "a quality judgment."
    )


# ============================================================
# VALIDATION & TRUST
# ============================================================

with tab_validation:
    st.markdown(
        """
This dashboard's recommendation engine has been checked against
real, held-out data at every stage -- not just fit to the data it
was built from. This section presents those real results directly,
including the ones that are modest rather than dramatic, since an
honest picture of what's actually been confirmed is more useful
than a purely favorable one.

## Placebo Tests

Several individual features were tested by comparing the real,
correctly-ordered version against a deliberately shuffled version
of the same data. If a feature carries genuine signal, the real
version should outperform its own scrambled copy -- if it doesn't,
the model may just be fitting noise.

| Feature | Real result | Placebo result |
|---|---|---|
| New-pitch distance | 87.5% sign consistency | 62.5% |
| Arsenal synergy (as a predictor) | 100% sign consistency | 58.3% |
| Stage 1 pitch-quality prediction | 0.448 real correlation | -0.122 |
| Swap usage-magnitude term | 0.0227 MAE | 0.0235 MAE |
        """
    )
    st.info(
        "**In plain terms:** imagine scrambling up which pitcher "
        "goes with which real outcome, completely at random, and "
        "seeing if the model still \"works\" -- if it does just as "
        "well on scrambled, meaningless data as on the real thing, "
        "that would mean it wasn't actually learning anything real "
        "in the first place. Every one of these four features passed "
        "that test: the real, correctly-matched version genuinely "
        "outperformed the randomly-scrambled version, confirming "
        "each one is picking up on something real, not coincidence. "
        "Stage 1 (predicting how good a brand-new pitch will be "
        "before a pitcher has even thrown it in a game) shows the "
        "clearest gap -- a real correlation of 0.448 vs. essentially "
        "zero (-0.122) for the scrambled version."
    )
    st.markdown(
        """
## Walk-Forward Validation

The strongest form of validation used in this project: the model
is trained ONLY on real events through a cutoff year, then tested
ONLY on real events from years it never saw during training --
ruling out the model simply memorizing patterns from the same pool
it's evaluated against.

| Model | Real MAE improvement over naive baseline | Ranking concordance |
|---|---|---|
| Single-change (ADD/DROP/USAGE) | ~5-9% | 64-71% |
| Swap predictions | ~37% | 78.9% |
        """
    )
    st.info(
        "**In plain terms:** this is the closest thing to a real "
        "test of \"does this actually predict the future,\" not just "
        "\"does it explain the past.\" The model was trained using "
        "only seasons through a certain year, then asked to predict "
        "outcomes for real pitchers and real seasons it had never "
        "seen at all -- like a true test rather than a practice run. "
        "**Ranking concordance** answers a very concrete question: "
        "if you give the model two different possible changes a "
        "pitcher could make, how often does it correctly pick which "
        "one will actually turn out better? A coin flip would get "
        "this right 50% of the time. The model gets it right "
        "64-71% of the time for single changes, and about 79% of "
        "the time for pitch swaps -- meaningfully, genuinely better "
        "than guessing. **MAE improvement** is about a different, "
        "harder question: not just \"which is better,\" but \"exactly "
        "how much better, in real numbers.\" Here the single-change "
        "models are only modestly better than just assuming nothing "
        "changes at all (~5-9%) -- so the honest, consistent finding "
        "across this whole project is that this system is "
        "considerably more trustworthy for \"which option should I "
        "pick\" than for \"exactly how much will this help.\""
    )
    st.markdown(
        """
## Direct Comparison Against the Previous System

Before this engine existed, this dashboard's own arsenal
recommendations came from a different, purely structural model
(PAOM's own recommendation logic). The two were directly compared
head to head, checking what actually happened when the previous
system's recommendations were followed by real pitchers:

| Change type | PAOM's real followed outcome vs. baseline |
|---|---|
| ADD | Worse (n=162 real events) |
| USAGE_INCREASE | Worse (n=82 real events) |
| USAGE_DECREASE | Worse (n=215 real events) |
        """
    )
    st.info(
        "**In plain terms:** the old system's advice was checked "
        "against what actually happened to real pitchers who "
        "followed it, compared to how similar pitchers who made any "
        "other kind of change did overall. Across every change type "
        "with enough real cases to check, pitchers who followed the "
        "old system's specific advice did WORSE on average, not "
        "better. That's a direct, real-world result, not just a "
        "difference of statistical opinion -- and it's the actual "
        "reason the old system's recommendations were dropped from "
        "this dashboard entirely, rather than kept as a second "
        "opinion alongside the new one."
    )
    st.markdown(
        """
## Confidence Calibration

A real, checked relationship: does more supporting evidence
actually correspond to more accurate predictions? Real historical
events were binned by evidence volume and compared against their
own real, walk-forward-validated error:

- Lowest-evidence bin: real MAE 0.0266
- Highest-evidence bin: real MAE 0.0240
        """
    )
    st.info(
        "**In plain terms:** this checks whether the confidence "
        "tiers you see on recommendations (Limited/Moderate/Strong) "
        "actually mean anything real, rather than being an "
        "arbitrary label. They do -- predictions backed by more real "
        "historical evidence really were somewhat more accurate on "
        "genuinely unseen data. That said, the real difference is "
        "modest (roughly 10%), not dramatic -- a \"Strong evidence\" "
        "recommendation is somewhat more trustworthy than a "
        "\"Limited evidence\" one, not vastly more so."
    )
