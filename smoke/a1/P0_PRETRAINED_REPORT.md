# A1 P0 pretrained baseline

P0 passed: one pretrained checkpoint reproduces standard-NMS and end-to-end NMS-free paths.

## Accuracy on COCO val2017

| Cell | Head path | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | one-to-many + NMS | 0.65900 | 0.51500 | 0.56200 | 0.40200 |
| B | one-to-one, NMS-free | 0.65600 | 0.50500 | 0.55000 | 0.39500 |

B - A mAP50-95: **-0.700 AP points**.

## Runtime route audit

| Cell | Detections | Real suppression kernel calls | Wrapper route |
| --- | ---: | ---: | --- |
| A | 77 | 16 | {"nms": 16} |
| B | 76 | 0 | {"end2end_score_filter": 16} |

## Batch-1 latency

| Device | A mean ± std ms | B mean ± std ms | A/B p50 ms | A/B p90 ms | B - A ms | B vs A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.289 ± 0.089 | 3.284 ± 0.098 | 3.260/3.252 | 3.446/3.477 | -0.005 | -0.14% |
| cpu | 20.246 ± 0.379 | 20.757 ± 0.569 | 20.169/20.794 | 20.557/21.412 | +0.511 | +2.52% |

## Limitations

- P0 compares two native inference paths in the same pretrained checkpoint; it is not the P1 MoE ablation.
- PyTorch runtime NMS tracing is separate from future ONNX graph auditing.
