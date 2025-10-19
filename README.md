# Ethereum Phishing detection using GNN

This project presents a machine learning approach to detect phishing accounts on the Ethereum blockchain. Utilizing a Graph Neural Network (GNN) built with PyTorch Geometric, the model learns to classify addresses by analyzing the structural and temporal patterns within their local transaction subgraphs. The pipeline involves engineering a 9-feature set from raw blockchain data, which is then cleaned using log transformation and global normalization, and the model is trained using a Focal Loss function to handle severe class imbalance. The final trained model demonstrates the ability to effectively identify malicious financial behavior, achieving a strong balance between precision and recall on a real-world dataset. Here is a detailed description on how I implemented this.

## 1: Data Collection and Processing

The success of the model relies on a robust and well-structured dataset. The pipeline for data collection and processing was executed in four main stages, transforming raw blockchain data into a model-ready format.

### 1. Labeled Address Collection

The initial dataset was formed by gathering a ground-truth list of both malicious and legitimate Ethereum addresses.
Phishing Addresses: A comprehensive list of malicious accounts was compiled from two primary sources to ensure both historical and recent data:

Historical Data: Parsed from the community-vetted addresses-darklist.json file in the ethereum-lists GitHub repository.

Recent Data: Scraped from the latest scam reports on Chainabuse, providing up-to-date threat intelligence.

Legitimate Addresses: A corresponding list of normal, non-malicious addresses was automatically sampled by a script that queried the Etherscan API for recent transactions and selected addresses with no known malicious labels.

### 2. Transaction History Fetching

With the labeled address lists, the full transaction history for each address was downloaded.

A Python script iterated through each address and used the Etherscan API to fetch its complete transaction history.

In line with the methodology from the source research paper, a filter was applied to ensure data quality. Any address with fewer than 5 or more than 1,500 transactions was discarded.

All valid transactions were consolidated into a single master file, all_transactions.csv, which served as the raw dataset for the next stage.

### 3. Graph Construction and Feature Engineering

The raw transaction log was then processed to create localized, feature-rich subgraphs.

For each "root" address from the initial lists, a 1-hop ego subgraph was constructed. This network consists of the root address and all other addresses it directly transacted with.

For every node within each subgraph, a set of 9 features was engineered by analyzing its transaction history to summarize its behavior:

* eth_sent: Total ETH sent.

* eth_received: Total ETH received.

* num_sent_txs: Total count of outgoing transactions.

* num_received_txs: Total count of incoming transactions.

* total_txs: Sum of all transactions.

* account_age_days: Time in days between the account's first and last transaction.

* avg_time_diff_hours: Average time between consecutive transactions.

* avg_tx_value: Average value of all its transactions.

* std_dev_tx_value: Standard deviation of its transaction values.

The structured data for each subgraph was saved into two separate files: _nodes.csv (containing the nodes and their 9 features) and _edges.csv (defining the transaction links).

### 4. Final Data Transformation

Before being fed to the model, the feature data underwent two crucial in-memory transformations to optimize it for training.

Log Transformation: To handle the skewed nature of financial data, a log1p transform was applied to features with a wide range of values (e.g., eth_sent, avg_tx_value). This compresses the feature space and reduces the impact of extreme outliers.

Global Normalization: A MinMaxScaler was fitted on the entire dataset to learn the global minimum and maximum for each feature. This pre-fitted scaler was then used to normalize all features to a consistent range (0 to 1), ensuring that the model treats all features with equal importance during training.

## 2. Model Architecture and Training Pipeline

The core of this project is an end-to-end machine learning pipeline designed to train a Graph Neural Network (GNN) on the processed transaction subgraphs. The architecture and training methodology were fine tuned to achieve optimal performance, with a focus on handling the severe class imbalance inherent in fraud detection datasets.

### 1. Model Architecture

A 4-layer GraphSAGE (SAGEConv) network was implemented using PyTorch Geometric. This architecture was chosen for its strong performance and scalability on inductive node classification tasks.

Graph Convolutional Layers: The model consists of four stacked SAGEConv layers, each with a hidden dimension of 256. These layers perform message passing, allowing each node to aggregate feature information from its local neighborhood. This hierarchical aggregation enables the model to learn complex structural patterns within the transaction graph.

Activation and Regularization: A ReLU (Rectified Linear Unit) activation function is applied after each convolutional layer to introduce non-linearity, allowing the model to learn more complex relationships. To prevent overfitting, a Dropout layer with a probability of p=0.5 is also applied after each activation.

Global Pooling and Readout: A key feature for achieving high performance was the implementation of a hybrid readout mechanism. After the final GNN layer, the model performs two operations:

* It extracts the specific, updated feature vector for the central root node of the subgraph.

* It uses a global_mean_pool layer to compute a single embedding that summarizes the average characteristics of the entire subgraph.
These two embeddings—the specific and the contextual—are then concatenated to form a richer, more informative feature vector for the final classification step.

Classifier: The final, combined feature vector is passed to a Linear layer which acts as the classifier, outputting the raw logits for the two classes ("legitimate" and "phishing").

### 2. Training Methodology

Data Handling: The custom PhishingSubgraphDataset class loads the processed data. The full dataset is split into training (70%), validation (15%), and test (15%) sets. Sampling strategy was used to ensure that the proportion of phishing-to-legitimate samples was consistent across all three splits.

Loss Function: To combat the severe class imbalance, a Focal Loss function was implemented. Unlike standard cross-entropy, Focal Loss dynamically down-weights the loss assigned to easy, well-classified examples (the abundant "legitimate" class). This forces the model to focus its learning efforts on the hard-to-classify, rare "phishing" examples, significantly improving recall and the overall F1 score.

Optimization: The model's parameters were optimized using the Adam optimizer with a learning rate of 0.0001. Training was conducted for 70 epochs, a duration found to be optimal for model convergence during experimentation.
