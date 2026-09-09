# A1 P2 limitations

This file records the scope of the P2 conclusions.

- The main detect-side interventions are fixed-seed, short-budget pilots. They
  establish controlled evidence and negative results, not universal rankings
  across seeds or training schedules.
- The corrected candidate-budget study used train5000/val512, five epochs,
  seed 260829, and the locked optimizer/freeze policy. Its `topk=10` result
  must not be generalized to every TAL or one-to-one assigner implementation.
- The fixed validation audit used each run's `last.pt`; comparisons with
  best-checkpoint or full-COCO results require the corresponding protocol.
- MoE latency measurements describe the current implementation and device
  path. They do not prove that all MoE implementations incur the same router
  or dispatch overhead.
- The `seg` extension was a single-seed pilot with a short schedule and
  randomly initialized mask/proto components. Its mask accuracy is not a
  cross-task claim about MoE quality.
- Pose was not run because the A1 task specifies `seg` or `pose` as an
  extension alternative, not a requirement to complete both.
- The evidence supports a constrained negative conclusion: under the locked
  protocols, no stable MoE or assigner/loss gain was demonstrated. It does not
  identify one universal root cause for the End-to-End accuracy gap.
