# Introduction

This is an implementation of IsolationForest that supports categorical and text features as well as numerical features.
The standard definition of IsolationForest [[1]](*1) considers only numerical attributes. In order to be able to train
an algorithm on a dataset containing both numerical and categorical features, one needs to convert categorical features
into numerical features using a technique called one hot encoding. Another drawback of this approach is when the
categorical feature to be encoded has a lot of possible values. This results in a large set of binary features. So if a
tree would randomly pick a subset of the features for splitting, it is more likely that those one-hot-encoded features
be picked up in comparison to the numerical features. This had led me to implement an IsolationForest that supports
categorical features without any modification on the original dataset.

The example file in the script's directory shows how to use this algorithm.

P.s.: For more information please refer to my master's thesis in the current project.

# IsolationForest with categorical features

### Subset of length 1[[2]](*2)

### Subset of variable length

# IsolationForest with textual features

## References

<a id="1">[1]</a>
Liu, Fei Tony; Ting, Kai Ming; Zhou, Zhi-Hua (December 2008). "Isolation Forest".

<a id="2">[2]</a>
Mathieu Garcherya, Michael Granitzer (September 2018). "On the influence of categorical features in ranking anomalies
using mixed data".
