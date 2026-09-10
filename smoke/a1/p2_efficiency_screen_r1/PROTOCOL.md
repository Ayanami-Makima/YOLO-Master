# P2 late-layer MoE efficiency screen

Only backbone layer 8 uses MoE; layers 4/6 retain dense residual factors. The six variants are two equal-compute Dense controls and four MoE variants (2/4 experts × Top-1/Top-2). All runs use the native pretrained C3k2 base, frozen BatchNorm, one pilot epoch, 5000/512 images, batch 4, SGD and seed 260829.

This is a screening pilot. It cannot replace the locked P1 factorial or support a long-training claim until a variant passes both precision and latency gates.
