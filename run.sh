#!/bin/bash
python src/multi_class_text_classifier/train_bert_based_centralized.py
python src/multi_class_text_classifier/train_bert_based_per_country.py
python src/multi_class_text_classifier/train_bert_based_federated.py
python src/multi_class_text_classifier/train_bert_based_federated_private.py

