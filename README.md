# ML4C: Seeing Causality Through Latent Vicinity

ML4C (Machine Learning for Causality) is a **supervised** causal discovery approach on **observational** data (and currently only supports **discrete** data). Starting from an input dataset with the corresponding skeleton provided, ML4C classifies (orients) whether each unshielded triple is a v-structure or not, and then outputs the corresponding CPDAG. Theoretically, ML4C is asymptotically correct by considering the graphical predicates in vicinity of each unshielded triple. Empirically, ML4C remarkably outperforms other state-of-the-art algorithms in terms of accuracy, reliability, robustness and tolerance. See our [paper](https://arxiv.org/abs/2110.00637) for more details.

---

## Basic Usage Example

```bash
cd Examples/
python main.py
```

This example orients a given skeleton. A simple call to `orient_skeleton` would work. Specifically, the arguments are:

+ `datapath`: `str`. Path to the observational data records. Should end with `.npy`, with the array in shape `(n_samples, n_variables)`. Now we only support discrete data, so the entries of this data array must be integers.
+ `skeletonpath`: `str`. Path to the provided skeleton's adjacency matrix. Should end with `.txt`, with the array in shape `(n_variables, n_variables)`. If `i--j` is in the skeleton, then `a[i, j] = a[j, i] = 1`, and otherwise `a[i, j] = a[j, i] = 0`. Note that, 
  + You may obtain the skeleton from data using standard algorithms (e.g., PC, GES, etc.), and then undirect all edges.
  + Or alternatively, you may try out our [ML4S](https://www.microsoft.com/en-us/research/uploads/prod/2022/07/ML4S-camera-ready.pdf) with [code](https://github.com/microsoft/reliableAI/tree/main/causal-kit/ML4S).
+ `savedir`: `str`. The directory to save the result CPDAG's adjacency matrix. If `a[i, j] = 1` and `a[j, i] = 0` then there is a directed edge `i->j`. If `a[i, j] = a[j, i] = 1` then there is an undirected edge `i--j`. Otherwise there is no edge between `i` and `j`.


## Classifier Training

In this repository, we provide a pre-trained classifier `./Learner/ML4C_learner.model`. To reproduce this classifier, you may run the following steps:

```bash
cd Learner/
python SynthesizeData.py      # Generate synthetic graph strutures and data records
python GenerateFeatures.py    # Generate features for each unshielded triple, based on the vicinity information
```

Then train a supervised classifier using whatever framework you like (e.g., here we use [XGBoost](https://xgboost.readthedocs.io/en/stable/)). Note that if you would like to customize your own classifier based on your own synthetic data (e.g., for continuous case), you may also follow the steps above.

---

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
