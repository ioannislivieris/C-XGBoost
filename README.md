# C-XGBoost: A tree boosting model for causal effect estimation

This is a repository for the implementation of Causal eXtreme Gradient Boosting (C-XGBoost) model, for causal effect estimation.


**Abstract:** Causal effect estimation aims at estimating the Average Treatment Effect as well as the Conditional Average Treatment Effect of a treatment to an outcome from the available data. This knowledge is important in many safety-critical domains, where it often needs to be extracted from observational data. In this work, we propose a new causal inference model, named C-XGBoost, for the prediction of potential outcomes. The motivation of our approach is to exploit the superiority of tree-based models for handling tabular data together with the notable property of causal inference neural network-based models to learn representations that are useful for estimating the outcome for both the treatment and non-treatment cases. The proposed model also inherits the considerable advantages of XGBoost model such as efficiently handling features with missing values requiring minimum preprocessing effort, as well as it is equipped with regularization techniques to avoid overfitting/bias. Furthermore, we propose a new loss function for efficiently training the proposed causal inference model. The experimental analysis, which is based on the performance profiles of Dolan and Moré as well as on post-hoc and non-parametric statistical tests, provide strong evidence about the effectiveness of the proposed approach.

**Keywords:** Causal inference, XGBoost, treatment effect estimation, potential outcomes

**Cite:** Kiriakidou, N., Livieris I.E. & Diou, C. (2024). [C-XGBoost: A tree boosting model for causal effect estimation](https://link.springer.com/chapter/10.1007/978-3-031-63219-8_5). IFIP International Conference on Artificial Intelligence Applications and Innovations.

<br/>

## Table of contents

- [How to run](#how-to-run)
- [Citation](#citation)
- [Contact](#mailbox-contact)

<br/>


## How to run

1. Create a virtual environment 
```
    python -m  venv .venv
```

2. Activate the virtual environment 
```
    source .venv/bin/activate
```
3. Install requirements 
```
    pip install -r requirements.txt
```
4. Run
```
    python c-xgboost.py
```

<br/>


## Citation
```
@inproceedings{kiriakidou2024c,
  title={C-XGBoost: A tree boosting model for causal effect estimation},
  author={Kiriakidou, Niki and Livieris, Ioannis E and Diou, Christos},
  booktitle={IFIP International Conference on Artificial Intelligence Applications and Innovations},
  pages={58--70},
  year={2024},
  organization={Springer}
}
```
<br/>

## :mailbox: Contact

Ioannis E. Livieris (livieris@unipi.gr)
