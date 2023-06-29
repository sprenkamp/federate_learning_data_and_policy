import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm 
import random

# Load df from CSV file
data="df"
DF = pd.read_csv(f'data/{data}.csv', on_bad_lines="skip")
max_length = 512
DF = DF[DF['x'].str.len() < max_length]
DF = DF[DF['y']!=-1]
DF = DF.dropna(subset=['y'])

model_name = 'bert-base-multilingual-uncased'
num_labels = len(DF.y.unique())
tokenizer = BertTokenizer.from_pretrained(model_name)

Q=0.5 # fraction of clients to be chosen
number_fed_rounds=3
noise_scale=0.1

def get_sample(dataset,k=0.1):
    return random.sample(dataset,int(k * len(dataset)))

def cal_sensitivity(lr, clip, dataset_size): #TODO cite this paper: [1] W. Yang et al., "Gain Without Pain: Offsetting DP-Injected Noises Stealthily in Cross-Device Federated Learning," in IEEE Internet of Things Journal, vol. 9, no. 22, pp. 22147-22157, 15 Nov.15, 2022, doi: 10.1109/JIOT.2021.3102030. 2] M. Hu et al., "AutoFL: A Bayesian Game Approach for Autonomous Client Participation in Federated Edge Learning," in IEEE Transactions on Mobile Computing, doi: 10.1109/TMC.2022.3227014. [3] Y. Zhou et al., "Optimizing the Numbers of Queries and Replies in Convex Federated Learning with Differential Privacy," in IEEE Transactions on Dependable and Secure Computing, doi: 10.1109/TDSC.2023.3234599.
    return 2 * lr * clip / dataset_size

def add_noise(net, sigma):
        # sensitivity = cal_sensitivity(lr=lr, clip=10, dataset_size=len(train_inputs))
        state_dict = net.state_dict()
        for k, v in state_dict.items():
            state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sigma, size=v.shape)).long().to(device) #gaussian noise

with open("models/results/results_federated_private.txt", "w") as file:
    overall_accuracy_lst = []
    overall_precision_lst = []
    overall_recall_lst = []
    overall_f1_lst = []
    for fed_round in range(1,number_fed_rounds+1):
        weight_list = []
        random_chosen_clients = get_sample(list(DF.federation_level.unique()),Q)
        print("random_chosen_clients", random_chosen_clients)
        for country in random_chosen_clients:
            df = DF[DF.federation_level==country]
            print(f"running fed round {fed_round} of {country} with {len(df)}")
            if fed_round==1:  
                model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels) # NOTE num_labels need to start at 0 important for labelling
                # Split the data into train, validation, and test sets
            else:
                model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
                sigma =  noise_scale * 2 / len(random_chosen_clients)
                add_noise(net=model, sigma=sigma)
                # Set the model's weights to the averaged weights
                model.load_state_dict(averaged_weights)

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

            # Tokenize and encode the text data for train, validation, and test sets
            train_inputs = tokenizer(
                train_data['x'].tolist(),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            val_inputs = tokenizer(
                val_data['x'].tolist(),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            test_inputs = tokenizer(
                test_data['x'].tolist(),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Convert labels to tensors for train, validation, and test sets
            train_labels = torch.tensor(train_data['y'].tolist())
            val_labels = torch.tensor(val_data['y'].tolist())
            test_labels = torch.tensor(test_data['y'].tolist())

            # Create TensorDatasets for train, validation, and test sets
            train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
            val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)
            test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

            # Define batch size and create DataLoaders for train, validation, and test sets
            batch_size = 8
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

            # Define optimizer and learning rate scheduler
            lr = 2e-5
            optimizer = AdamW(model.parameters(), lr=lr)
            total_steps = len(train_dataloader) * 10  # 10 epochs
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            # Fine-tuning loop
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # device = "mps"
            # print(device)
            model.to(device)

            best_val_loss = float('inf')
            best_model = None

            for epoch in tqdm(range(10)):  # 10 epochs
                total_loss = 0
                model.train()

                for batch in train_dataloader:
                    input_ids, attention_mask, batch_labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    batch_labels = batch_labels.to(device)

                    model.zero_grad()

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=batch_labels
                    )

                    loss = outputs.loss
                    total_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) # gradient clipping with gaussian norm #TODO check documentation
                    optimizer.step()

                scheduler.step()

                # Calculate average training loss for the epoch
                avg_loss = total_loss / len(train_dataloader)
                # print(f'Training Loss: {avg_loss}')

                # Evaluate on the validation set
                model.eval()
                val_loss = 0
                val_correct = 0

                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids, attention_mask, batch_labels = batch
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        batch_labels = batch_labels.to(device)

                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=batch_labels
                        )

                        loss = outputs.loss
                        logits = outputs.logits

                        val_loss += loss.item()

                        # Calculate the number of correct predictions
                        _, predicted_labels = torch.max(logits, dim=1)
                        val_correct += torch.sum(predicted_labels == batch_labels).item()

                avg_val_loss = val_loss / len(val_dataloader)
                val_accuracy = val_correct / len(val_dataset)

                # print(f'Validation Loss: {avg_val_loss}')
                # print(f'Validation Accuracy: {val_accuracy}')

                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model = model.state_dict()

            # Load the best model
            model.load_state_dict(best_model)

            # # Evaluate the model on the test set
            # model.eval()
            # total_test_loss = 0
            # total_test_correct = 0

            # with torch.no_grad():
            #     for batch in test_dataloader:
            #         input_ids, attention_mask, batch_labels = batch
            #         input_ids = input_ids.to(device)
            #         attention_mask = attention_mask.to(device)
            #         batch_labels = batch_labels.to(device)

            #         outputs = model(
            #             input_ids=input_ids,
            #             attention_mask=attention_mask,
            #             labels=batch_labels
            #         )

            #         loss = outputs.loss
            #         logits = outputs.logits

            #         total_test_loss += loss.item()

            #         # Calculate the number of correct predictions
            #         _, predicted_labels = torch.max(logits, dim=1)
            #         total_test_correct += torch.sum(predicted_labels == batch_labels).item()

            # # Calculate average test loss and accuracy
            # avg_test_loss = total_test_loss / len(test_dataloader)
            # accuracy = total_test_correct / len(test_dataset)

            # # # print(f'Test Loss: {avg_test_loss}')
            # # file.write(f'Test Accuracy round {fed_round} {country}: {accuracy} \n')
            
            weight_list.append(model.state_dict())
        
        # print(weight_list)
        averaged_weights = {}

        # Average the weights
        for key in weight_list[0].keys():
            weights_float = [weights[key].float() for weights in weight_list]
            averaged_weights[key] = torch.mean(torch.stack(weights_float), dim=0)

    # run federated model on country test sets
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    sigma =  noise_scale * 2 / len(random_chosen_clients)
    add_noise(net=model, sigma=sigma)
    # Set the model's weights to the averaged weights
    model.load_state_dict(averaged_weights)
    for country in DF.federation_level.unique():
            df = DF[DF.federation_level==country]
            print(f"running final inference of {country} with {len(df)}")
            # Split data into train, validation, and test sets
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

            # Tokenize and encode the text data for train, validation, and test sets
            train_inputs = tokenizer(
                train_data['x'].tolist(),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            val_inputs = tokenizer(
                val_data['x'].tolist(),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            test_inputs = tokenizer(
                test_data['x'].tolist(),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Convert labels to tensors for train, validation, and test sets
            train_labels = torch.tensor(train_data['y'].tolist())
            val_labels = torch.tensor(val_data['y'].tolist())
            test_labels = torch.tensor(test_data['y'].tolist())

            # Create TensorDatasets for train, validation, and test sets
            train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
            val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)
            test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

            # Define batch size and create DataLoaders for train, validation, and test sets
            batch_size = 8
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

            # Define optimizer and learning rate scheduler
            lr = 2e-5
            optimizer = AdamW(model.parameters(), lr=lr)
            total_steps = len(train_dataloader) * 10  # 10 epochs
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            # Fine-tuning loop
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # device = "mps"
            # print(device)
            model.to(device)

            # Evaluate the model on the test set
            model.eval()
            total_test_loss = 0
            total_test_correct = 0
            predicted_labels_all = []
            true_labels_all = []

            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids, attention_mask, batch_labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=batch_labels
                    )

                    loss = outputs.loss
                    logits = outputs.logits

                    total_test_loss += loss.item()

                    # Calculate the number of correct predictions
                    _, predicted_labels = torch.max(logits, dim=1)
                    total_test_correct += torch.sum(predicted_labels == batch_labels).item()

                    # Collect predicted and true labels for metric calculation
                    predicted_labels_list = predicted_labels.detach().cpu().tolist()
                    batch_labels_list = batch_labels.detach().cpu().tolist()
                    predicted_labels_all.extend(predicted_labels_list)
                    true_labels_all.extend(batch_labels_list)

                # Calculate average test loss, accuracy, precision, recall, and F1 score
                # avg_test_loss = total_test_loss / len(test_dataloader)
                accuracy = accuracy_score(true_labels_all, predicted_labels_all)
                precision = precision_score(true_labels_all, predicted_labels_all, average='macro')
                recall = recall_score(true_labels_all, predicted_labels_all, average='macro')
                f1 = f1_score(true_labels_all, predicted_labels_all, average='macro')
                # file.write(f'Test Loss {country}: {avg_test_loss} \n')
                file.write(f'Test Accuracy {country}: {accuracy} \n')
                file.write(f'Test Precision {country}: {precision} \n')
                file.write(f'Test Recall {country}: {recall} \n')
                file.write(f'Test F1 {country}: {f1} \n')
                overall_accuracy_lst.append(accuracy)
                overall_precision_lst.append(precision)
                overall_recall_lst.append(recall)
                overall_f1_lst.append(f1)


    file.write(f'Overall Accuracy: {np.mean(overall_accuracy_lst)} \n')
    file.write(f'Overall Precision: {np.mean(overall_precision_lst)} \n')
    file.write(f'Overall Recall: {np.mean(overall_recall_lst)} \n')
    file.write(f'Overall F1: {np.mean(overall_f1_lst)} \n')
    model.save_pretrained(f'models/federated')