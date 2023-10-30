# Federated Learning Experiments
The repository accompanies our publication in the Data and Policy Journal titled "Overcoming Intergovernmental Data Sharing Challenges with Federated Learning". The repository contains a series of experiments around federated learning, primarily leveraging BERT for multi-class text classification tasks. Our main script `run.sh` runs a set of experiments: centralized learning, per-country learning, federated learning, and privacy-preserving federated learning. Please contact the repo owner to obtain the used dataset, as we do not share it due to privacy and ethical reasons.

## Installation
1. Clone the repository
```bash
git clone https://github.com/sprenkamp/federated_learning_data_and_policy.git
cd federated_learning_data_and_policy
```

2. Create a Python virtual environment and activate it (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
```

3. Install the required Python packages
```bash
pip install -r requirements.txt
```

## Running the Experiments
To run the experiments, use the `run.sh` script. Ensure that it has the right permissions.

```bash
bash run.sh
```

The `run.sh` script will run the following experiments in order:

1. **Centralized Learning:** This experiment uses BERT for multi-class text classification in a centralized manner.
```bash
python src/multi_class_text_classifier/train_bert_based_centralized.py
```

2. **Per-country Learning:** This experiment takes a look at learning BERT-based multi-class text classifiers per country.
```bash
python src/multi_class_text_classifier/train_bert_based_per_country.py
```

3. **Federated Learning:** In this experiment, we use BERT for multi-class text classification in a federated learning setup.
```bash
python src/multi_class_text_classifier/train_bert_based_federated.py
```

4. **Privacy-Preserving Federated Learning:** Lastly, this experiment adds a privacy-preserving layer to the federated learning setup.
```bash
python src/multi_class_text_classifier/train_bert_based_federated_private.py
```
