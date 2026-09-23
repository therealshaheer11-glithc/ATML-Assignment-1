# Task 2 v4 adoption decision

The v4 adversarial-input normalization rule is adopted for DANN and
CDAN using source information only.

- V3 DANN source macro-F1: 0.6049088768
- V4 DANN source macro-F1: 0.9408741433
- Source F1 change: 0.3359652665
- V3 maximum pre-clipping gradient norm: 40455498.9988
- V4 maximum pre-clipping gradient norm: 11.7314
- V3 clipped-step fraction: 0.934043
- V4 clipped-step fraction: 0.015502
- All seven source classes predicted by selected v4 checkpoint: yes
- Target labels accessed: no
- Training-code commit: `d26997b22d3b7722e2ecc828dde1445244afc04b`

The classifier continues to use raw features. Only the DANN/CDAN
domain-discriminator input uses per-example L2-normalized features.
All other locked settings remain unchanged.
