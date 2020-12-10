# Next-Basket-Recommendations
This repository contains my research work on building the state of the art next basket recommendations using techniques such as Autoencoders, TF-IDF, Attention based BI-LSTM and Transformer Networks.

This project has been tested with UCI's Online Retail Data Set I and II from the following locations:
1. https://archive.ics.uci.edu/ml/datasets/online+retail
2. https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

BACKGROUND:
This project builds up on ideas presented in the paper titled: Pre-training of Context-aware Item Representation for Next Basket Recommendation by Jingxuan Yang et al.
In my model, I generate binary predictions at the customer-product level based on resell and cross-sell combinations to predict whether a particular combination was bought in the next customer's transaction. Products are at the SKU level, which is the most granular level in the item hierarchy.

FEATURES:
1. Weekly sequence of quantities purchased of item I which are fed into attention based Bi-LSTM.
2. Weekly sequence of quantities returned of item I which are fed into attention based Bi-LSTM.
3. Compressed latent features from TF-IDF sparse matrix which are generated through a separate 3 layered autoencoder model.
4. Multi-headed attention based transformer layer on products bought together in each of the last 3 transactions of all customers.
5. Item embeddings based on customer-item combination.

MEMORY and COMPUTATION:
For batch predictions, generating customer-item combinations can be computationally intensive. However, there are ways to get around this problem:
1. While resell customer-product combinations are typically more predictive and important for retail transactions, cross sell customer-product combinations are more flexible and can be filtered out while building the model. The filtering could be done based on how popular or rare an item is, which items the business would like to promote, items that are similar to other items, etc.
2. While the model is in Python, it can be scaled up to run faster if the model universe and deep learning model leverages Spark or another distributed computing framework.
3. Although the model has been built on a sample using a CPU 32GB machine, the code can be modified to run on a multi-gpu machine such as AWS P2.8X large. 
These changes would speed up computations by more than 150X.

EVALUATION:
Currently the evaluation metrics include hit rates for top 10 products recommended for learn and validation datasets.
However AUC, recall@k, Mrr@k (mean reciprocal rank), Map@k (mean average precision) and Ndcg@K (normalized discounted cumulative gain) are some other metrics that can be added to Evaluation.py to evaluate the model's performance.

TO RUN:
Fork this repository and change the parameters in configs before calling main.py
