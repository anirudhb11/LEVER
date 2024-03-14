<img src="Resources/ICLR_Logo.jpg" height="100" align="right"/>
# LEVER: Enhancing Tail Performance In Extreme Classifiers by Label Variance Reduction
This is the official codebase for [ICLR 2024](https://iclr.cc/Conferences/2024) paper [ Enhancing Tail Performance In Extreme Classifiers by Label Variance Reduction](https://openreview.net/forum?id=6ARlSgun7J)

## Overview

Extreme Classification ([`XC`](http://manikvarma.org/downloads/XC/XMLRepository.html)) architectures, which utilize a massive one-vs-all classifier layer at the output, have demonstrated remarkable performance on problems with large label sets. However, these architectures are inaccurate on tail labels with few representative samples. This work explores the impact of label variance, a previously unexamined factor, on the tail performance
in extreme classifiers. It presents a method to systematically reduce label variance in XC by effectively utilizing the capabilities of an additional, tail-robust teacher model. It proposes a principled knowledge distillation framework, LEVER, which enhances tail performance in extreme classifiers with formal guarantees on generalization. Comprehensive experiments show that LEVER can enhance tail performance by around 5% and 6% points in PSP and coverage metrics, respectively, when combined with leading extreme classifiers. Moreover, it establishes a new state-of-
the-art when added to the top-performing [Ren ́ee](https://github.com/microsoft/renee) classifier.

## Cite

```bib
@inproceedings{
buvanesh2024enhancing,
title={Enhancing Tail Performance in Extreme Classifiers by Label Variance Reduction},
author={Anirudh Buvanesh and Rahul Chand and Jatin Prakash and Bhawna Paliwal and Mudit Dhawan and Neelabh Madan and Deepesh Hada and Vidit Jain and SONU MEHTA and Yashoteja Prabhu and Manish Gupta and Ramachandran Ramjee and Manik Varma},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=6ARlSgun7J}
}
```

## You May Also Like
- [DEXA: Deep Encoders with Auxiliary Parameters for Extreme Classification](https://github.com/Extreme-classification/dexa)
- [NGAME: Negative mining-aware mini-batching for extreme classification](https://github.com/Extreme-classification/ngame)
- [SiameseXML: Siamese networks meet extreme classifiers with 100M labels](https://github.com/Extreme-classification/siamesexml)
- [DeepXML: A Deep Extreme Multi-Label Learning Framework Applied to Short Text Documents](https://github.com/Extreme-classification/deepxml)
- [DECAF: Deep Extreme Classification with Label Features](https://github.com/Extreme-classification/DECAF)
- [ECLARE: Extreme Classification with Label Graph Correlations](https://github.com/Extreme-classification/ECLARE)
- [GalaXC: Graph Neural Networks with Labelwise Attention for Extreme Classification](https://github.com/Extreme-classification/GalaXC)

