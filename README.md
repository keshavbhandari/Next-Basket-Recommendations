# Next-Basket-Recommendations
This repository contains my research work on building the state of the art next basket recommendations using techniques such as Autoencoders, TF-IDF, Attention based BI-LSTM and Transformer Networks.

This project has been tested with UCI's Online Retail Data Set I and II from the following locations:
1. https://archive.ics.uci.edu/ml/datasets/online+retail
2. https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

BACKGROUND:
This project builds up on ideas presented in the paper titled: Pre-training of Context-aware Item Representation for Next Basket Recommendation by Jingxuan Yang et al.
In my model, I generate binary predictions at a customer-product level based on resell and cross-sell combinations. Products are at SKU level (most granular).
While resell customer-product combinations are important, cross sell customer-product combinations are flexible and can be filtered based on important product items to a business. These could be based on how popular or rare an item is, which items the business would like to promote, etc.
Although the model is in Python, it can be scaled up if the model universe and deep learning model leverages Spark or another distributed computing framework.

EVALUATION:
Currently the evaluation metrics include hit rates for top 10 products recommended for learn and validation datasets.
However AUC, recall@k, Mrr@k (mean reciprocal rank), Map@k (mean average precision) and Ndcg@K (normalized discounted cumulative gain) are some other metrics that can be added to evaluate the model's performance.

TO RUN:
Fork this repository and change the parameters in configs before calling main.py
