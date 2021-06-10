# modeling_citations

## Motivation
This project is designed to recreate the bias in citation practices in neuroscience reported [here](https://www.nature.com/articles/s41593-020-0658-y). Here,
the perceived gender of authors is inferred by comparing the first names of authors to a [database of names from multiple countries](https://gender-api.com/).
It is important to note the aim is the understand the bias associated with the perceived gender of a name, not the individuals true gender identity or sex. 
This work shows (1) the men-led teams are cited more than women-led teams given their presence in the field (2) that these disparities are increasing over time, 
despite the field becomng more diverse, and (3) that these disparities are driven mostly by the majority (men-led teams). Here, we seek to gain an understanding of 
the possible drivers of and most effective interventions for this phenomenon through simulations of citation practices.

We define artificial agents (authors) that have different mental estimates of the network of people that make up the field of neuroscience. In this network,
nodes are authors and edges are co-authorships. Citation lists are generated from biased random walks on these networks. These estimated networks can be 
influenced by meetings with other authors. Estimated networks will also periodically prune (or forget) authors who do not frequenctly appear in citation lists.

## Dependencies
This project is done entirely in Python (v>=3.6), and the package versions used are given in the requirements.txt file

## Code
### Classes
1. `Author`: defined in `functions/author_fns.py`. This class defines the properties and functions of 'authors' in the simulation. Each author has a gender, 
              and 5 parameters that determine how it interacts with other authors.

### Key Functions
1. `Author.init_network()`: defined in `functions/author_fns.py`.This function intiaites an Authors estimate of the field, using a Levy flight on the true co-author 
                            network/ Can be biased towards authors of one gender with `net_bias`
2. `Author.get_cites()`: defined in `functions/author_fns.py`. Generates a citation list from a random walk an the authors network. Can be biased towards one of 
                          the genders with `walk_bias`
3. `Author.forget()`: defined in `functions/author_fns.py`. Removed some number of authors that have not been viewed recently. The number is determined by `forget_bias`
4. `compare_nets()`: defined in `functions/author_fns.py`. Decides whether two authors will mett with and learn from each other based on their `meet_bias` parameters

### Scripts
1. `eda_preproc.ipynb`: this script shows basic statistics (degree distribution, number of nodes, etc.) for the neuroscience co-author network used in this research
2. `class_fn_test.ipynb`: this script tests failure cases for all functions, and plots how basic statics (network size, gender distribution) change for different
                          values of bias parameters
3. `simulations.ipynb`: this script simulates meetings between authors, and plots citation practices over simulations
