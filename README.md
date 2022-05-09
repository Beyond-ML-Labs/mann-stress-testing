# MANN Stress Testing

This repository contains a set of experiments conducted in a meta-study surrounding MANN. The following studies are conducted:

- How many tasks?
  - For this study, experiments are performed to test how many individual tasks can be "fit" into a MANN model given a one-shot sparsification technique vs an iterative pruning technique.
- Better Generalization?
  - This study aims to identify whether pruned models are able to generalize/draw conclusions better than traditionally-trained models. This will be tested across a couple of use cases and pruning techniques across learning and testing scenarios. We will test whether less data (both in total and on a per-class basis) can be supplied to already-pruned, sequentially-pruned, and unpruned models and see how performance is affected.