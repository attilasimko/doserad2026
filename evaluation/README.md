# Challenge evaluation suite

`doserad2026_evaluator/` is the **DoseRAD2026 challenge's own** evaluation code,
not part of this submission. It is included so the scoring in this repository is
the scoring the challenge applies, rather than a reimplementation of it.

`eval_patient_total.py`, `eval_validation.py` and `eval_plans.py` import
`masked_beam_mae` and `idd_curve_distance` from it.

`GROUND_TRUTH_FORMAT.md` documents the layout the evaluator expects.

If you are redistributing this repository, check the challenge's terms for this
code — it is third party and carries no licence header here. It is not covered
by the repository's Apache-2.0 licence.
