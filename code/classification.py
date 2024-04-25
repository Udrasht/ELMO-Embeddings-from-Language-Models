import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchtext import vocab
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import random
import nltk
from collections import Counter
from torch.utils.data import Dataset
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torch import nn, optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
import os
from torchtext.vocab import build_vocab_from_iterator

nltk.download('punkt')

# Determine device (GPU if available, otherwise CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(DEVICE)



def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())
    
    # List of prepositions to remove (customize as needed)
    words_to_remove = ['pandora', 'passions', 'nix', 'spoilers', 'joking', 'draconian']

    prepositions = ['for', 'with', 'to', 'from', 'in', 'on', 'at', 'by']
    
    # Filter out prepositions and specific characters from words
    filtered_words = [word for word in words if word not in prepositions and word.isalnum()]
    
    # Remove words from the words_to_remove list
    filtered_words = [word for word in filtered_words if word not in words_to_remove]
    
    # Join filtered words back into a cleaned text string
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text











def clean_dataframe(df, column_name):
    # Apply preprocess_text function to the specified column in the DataFrame
    df[column_name] = df[column_name].apply(preprocess_text)
    
    return df

# Example usage:
filepath = "../data/train.csv"
num_rows_to_read = 120000

df = pd.read_csv(filepath, nrows=num_rows_to_read)

# Specify the column containing text to be cleaned
text_column = 'Description'

# Clean the DataFrame by removing prepositions and specific characters
cleaned_df_train = clean_dataframe(df, text_column)

# Display the cleaned DataFrame
# print(cleaned_df_train.head())


filepath_test = "../data/test.csv"


df_test = pd.read_csv(filepath_test)

# Specify the column containing text to be cleaned
text_column = 'Description'

# Clean the DataFrame by removing prepositions and specific characters
cleaned_df_test = clean_dataframe(df_test, text_column)

print("csv files load compleat")


class ProcessYelp():
  def __init__(self, cleaned_df, min_freq):
    # self.filepath = filepath
    self.min_freq = min_freq

    df = cleaned_df
    total_words = []
    for i in tqdm(range(len(df)), desc="Vocabulary"):
        line = df['Description'][i]
        total_words += [[word.lower()] for word in word_tokenize(line)]

    self.vocab = vocab.build_vocab_from_iterator(total_words,
                                                  min_freq = min_freq,
                                                  specials = ['<UNK>', '<PAD>'])
    self.vocab.set_default_index(self.vocab['<UNK>'])



        # Build vocabulary using torchtext's build_vocab_from_iterator with custom filter
  

    
  def save_vocab_to_json(self, filepath):
    # Ensure the directory exists
    vocab_tokens = self.vocab.get_itos()

        # Save vocabulary tokens to a text file
    with open(filepath, 'w', encoding='utf-8') as file:
        for token in vocab_tokens:
            file.write(token + '\n')
 



class LabelData(Dataset):
    def __init__(self, vocab, cleaned_df, max_length=35):
        self.vocab = vocab
        self.data = cleaned_df
        self.max_length = max_length

    def __getitem__(self, index):
        description = self.data.loc[index, 'Description']
        label = self.data.loc[index, 'Class Index']

        # Tokenize the description
        tokens = word_tokenize(description.lower())
        
        # Convert tokens to indices using vocab, and pad if necessary
        token_indices = [self.vocab[token.lower()] if token.lower() in self.vocab else self.vocab['<PAD>'] for token in tokens[:self.max_length]]
        token_indices += [self.vocab['<PAD>']] * (self.max_length - len(token_indices))

        return torch.tensor(token_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)


class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, embeddings):
        super(ELMo, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)

        self.layer_1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.layer_2 = nn.LSTM(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, X):
        embeddings = self.embedding(X)

        lstm1_output, _ = self.layer_1(embeddings)

        lstm2_output, _ = self.layer_2(lstm1_output)
        lstm2_output = self.dropout(lstm2_output)

        output = self.linear(lstm2_output)
        output = torch.transpose(output, 1, 2)
        return output


class Downstream(nn.Module):
    def __init__(self, embedding_size):
        super(Downstream, self).__init__()

        self.s1 = nn.Parameter(torch.ones(1))
        self.s2 = nn.Parameter(torch.ones(1))
        self.s3 = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.ones(1))

        # Change the output layer to 4 classes
        self.linear = nn.Linear(embedding_size, 4)  # Output layer with 4 units for 4 classes

    def forward(self, sentence):
        embeddings = elmo_bilstm.embedding(sentence)
        out_1, _ = elmo_bilstm.layer_1(embeddings)
        out_2, _ = elmo_bilstm.layer_2(out_1)
        

        s_sum = self.s1 + self.s2 + self.s3

        output = self.alpha * (
            self.s1 / s_sum * embeddings
            + self.s2 / s_sum * out_1
            + self.s3 / s_sum * out_2
        ).to(torch.float32)
        aggregated_output = torch.mean(output, dim=1)

        output = self.linear(aggregated_output)  # Apply linear transformation
        return output 
    
class Downstream(nn.Module):
    def __init__(self, embedding_size):
        super(Downstream, self).__init__()

        self.s1 = nn.Parameter(torch.ones(1))
        self.s2 = nn.Parameter(torch.ones(1))
        self.s3 = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.ones(1))

        # Change the output layer to 4 classes
        self.linear = nn.Linear(embedding_size, 4)  # Output layer with 4 units for 4 classes

    def forward(self, sentence):
        embeddings = elmo_bilstm.embedding(sentence)
        out_1, _ = elmo_bilstm.layer_1(embeddings)
        out_2, _ = elmo_bilstm.layer_2(out_1)
        

        s_sum = self.s1 + self.s2 + self.s3

        output = self.alpha * (
            self.s1 / s_sum * embeddings
            + self.s2 / s_sum * out_1
            + self.s3 / s_sum * out_2
        ).to(torch.float32)
        aggregated_output = torch.mean(output, dim=1)

        output = self.linear(aggregated_output)  # Apply linear transformation
        return output 



class DownstreamFrozenLambdas(nn.Module):
    def __init__(self, embedding_size):
        super(DownstreamFrozenLambdas, self).__init__()

        # Lambda parameters (frozen)
        self.s1 = nn.Parameter(torch.rand(1) * 0.5 + 0.5, requires_grad=False)  # Random value between 0.5 and 1
        self.s2 = nn.Parameter(torch.rand(1) * 0.5 + 0.5, requires_grad=False)  # Random value between 0.5 and 1
        self.s3 = nn.Parameter(torch.rand(1) * 0.5 + 0.5, requires_grad=False)  # Random value between 0.5 and 1
        self.alpha = nn.Parameter(torch.rand(1) * 0.5 + 0.5, requires_grad=False)  # Random value between 0.5 and 1

        # Output layer
        self.linear = nn.Linear(embedding_size, 4)  # Output layer with 4 units for 4 classes

    def forward(self, sentence):
        # Similar forward pass as above
        embeddings = elmo_bilstm.embedding(sentence)
        out_1, _ = elmo_bilstm.layer_1(embeddings)
        out_2, _ = elmo_bilstm.layer_2(out_1)

        # Combine word representations using frozen lambda weights
        s_sum = self.s1 + self.s2 + self.s3
        output = self.alpha * (
            self.s1 / s_sum * embeddings +
            self.s2 / s_sum * out_1 +
            self.s3 / s_sum * out_2
        ).to(torch.float32)

        # Aggregate output
        aggregated_output = torch.mean(output, dim=1)

        # Apply linear transformation
        output = self.linear(aggregated_output)

        return output
    
class LearnableFunction(nn.Module):
    def __init__(self, input_size, output_size):
        super(LearnableFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class DownstreamLearnable(nn.Module):
    def __init__(self, embedding_size):
        super(DownstreamLearnable, self).__init__()

        # Define the learnable function to combine word representations
        self.learnable_function = LearnableFunction(3 * embedding_size, embedding_size)

        # Other components of your model
        self.embedding_size = embedding_size
        self.linear = nn.Linear(embedding_size, 4)  # Output layer with 4 units for classification

    def forward(self, sentence):
        # Implement your forward pass using the learnable function and other components
        embeddings = elmo_bilstm.embedding(sentence)
        out_1, _ = elmo_bilstm.layer_1(embeddings)
        out_2, _ = elmo_bilstm.layer_2(out_1)

        # Combine word representations using the learnable function
        combined_representation = torch.cat([embeddings, out_1, out_2], dim=2).to(torch.float32)
        learned_embedding = self.learnable_function(combined_representation)
        learned_embedding=torch.mean(learned_embedding, dim=1)

        # Apply linear transformation
        output = self.linear(learned_embedding)
        return output


def train_model(model, train_dataloader, criterion, optimizer, epochs, device):
    train_losses = []

    # Training loop
    for epoch in tqdm(range(epochs)):
        total_loss_train = 0

        model.train()  # Set the model to training mode
        for i, (sentence, label) in enumerate(train_dataloader):
            sentence, label = sentence.to(device), label.to(device)
            label -= 1  # Adjust labels (assuming they are 1-based)

            optimizer.zero_grad()
            outputs = model(sentence)

            # Calculate loss
            batch_loss = criterion(outputs, label)
            total_loss_train += batch_loss.item()

            # Backpropagation
            batch_loss.backward()
            optimizer.step()

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss_train / len(train_dataloader)
        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

    # Plot the training loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model(downstream_model, dataloader_test, device):
    y_true = []
    y_pred = []

    downstream_model.eval()

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader_test):
            model_input = inputs.to(device)
            outputs = downstream_model(model_input)

            targets = targets.tolist()

            y_true += targets
            y_pred += (torch.argmax(outputs, dim=1).tolist())

    # Adjust y_pred to align with the range of y_true (subtract 1 from each element)
    y_pred_adjusted = [pred + 1 for pred in y_pred]

    print("Classification Report:")
    print(classification_report(y_true, y_pred_adjusted))

    return y_true, y_pred_adjusted



processed_data_train = ProcessYelp(cleaned_df_train, 5)

vocab_json_path = 'D:/Sem3/NLP/ELMO/code/vocab.json'

# Save the vocab to a JSON file

# processed_data_train.save_vocab_to_json(vocab_json_path)



vocab_tokens = processed_data_train.vocab.get_itos()

BATCH_SIZE=64
embedding_dim=300
learning_rate=0.001
hidden_size=50
dropout=0.1
epochs=10

label_train_dataset = LabelData(processed_data_train.vocab,cleaned_df_train,35)
train_dataloader_train = DataLoader(label_train_dataset, batch_size=BATCH_SIZE, shuffle=False)




label_test_dataset = LabelData(processed_data_train.vocab,cleaned_df_test,35)
dataloader_test = DataLoader(label_test_dataset, batch_size=1, shuffle=False)

embed_matrix = np.zeros((len(processed_data_train.vocab), embedding_dim))

embed_matrix_torch = torch.from_numpy(embed_matrix)
embed_matrix_torch = embed_matrix_torch.to(DEVICE)


print("embed_matrix.shape:",embed_matrix.shape)



model_path = '../models/bilstm.pt'

# Create an instance of your model
elmo_bilstm = (ELMo(len(processed_data_train.vocab), embedding_dim, 150, dropout, embed_matrix_torch).double().to(DEVICE)) # Instantiate your model class

# Load the model's state dictionary from the saved file
elmo_bilstm.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# downstream_frozen_model  = DownstreamFrozenLambdas(embedding_dim).to(DEVICE)
# criterion = nn.CrossEntropyLoss().to(DEVICE)  # Use CrossEntropyLoss for multi-class classification
# # Define optimizer and criterion
# optimizer_frozen  = optim.Adam(filter(lambda p: p.requires_grad, downstream_frozen_model .parameters()), lr=learning_rate)


# downstream_model= Downstream(embedding_dim).to(DEVICE)
# optimizer = optim.Adam(downstream_model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss().to(DEVICE)



downstream_model_Learnable = DownstreamLearnable(embedding_dim).to(DEVICE)
optimizer_Learnable = optim.Adam(downstream_model_Learnable.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(DEVICE)

classifer_model_path = '../models/classifier.pt'

downstream_model_Learnable.load_state_dict(torch.load(classifer_model_path, map_location=torch.device('cpu')))

# for train the model uncomment the below line and comment the above line

# train_model(downstream_model_Learnable, train_dataloader_train, criterion, optimizer_Learnable, epochs, DEVICE)


# for save the model


# output_path = 'classifier.pt'
# torch.save(downstream_model_Learnable.state_dict(), output_path)

y_true, y_pred = evaluate_model(downstream_model_Learnable, dataloader_test, DEVICE)

y_true_train, y_pred_train = evaluate_model(downstream_model_Learnable, train_dataloader_train, DEVICE)



accuracy_test = accuracy_score(y_true, y_pred)

print(f"Overall Accuracy: {accuracy_test}")

accuracy_train = accuracy_score(y_true_train, y_pred_train)

print(f"Overall Accuracy: {accuracy_train}")

def plot_confusion_matrix_and_classification_report(true_labels, predicted_labels, class_names, test_train):
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2)  # Adjust font scale
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix'+test_train)
    plt.show()

    # Plot the classification report
    plt.figure(figsize=(12, 10))
    report = classification_report(
        true_labels, predicted_labels, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    sns.heatmap(df.iloc[:-1, :].T, annot=True, cmap="YlGnBu", cbar=False)
    plt.title('Classification Report'+test_train)
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.show()

class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']


# for plot find the confussion matrix
# plot_confusion_matrix_and_classification_report(y_true, y_pred, class_names, "Test")

# plot_confusion_matrix_and_classification_report(y_true_train, y_pred_train, class_names, "Train")






def predict_label_input(input_sentence, downstream_model, vocab, DEVICE, max_length=35):
    # Define column names
    column_names = ["Description", "Class Index"]
    class_index = 1  # Assuming a default class index for the input sentence

    # Create a DataFrame with the input sentence and class index
    data = {column_names[0]: [input_sentence], column_names[1]: [class_index]}
    df_input = pd.DataFrame(data, columns=column_names)

    # Create a LabelData object for the input DataFrame
    label_input_dataset = LabelData(vocab, df_input, max_length)
    dataloader_input = DataLoader(label_input_dataset, batch_size=1, shuffle=False)

    y_pred = []

    downstream_model.eval()

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader_input, desc="Predicting Labels"):
            model_input = inputs.to(DEVICE)
            outputs = downstream_model(model_input)

            targets = targets.tolist()
            y_pred += (torch.argmax(outputs, dim=1).tolist())

    return y_pred

# Example usage:
input_sentence = input("Enter sentence: ")
predicted_labels = predict_label_input(input_sentence, downstream_model_Learnable, processed_data_train.vocab, DEVICE)
print("Predicted Labels:", predicted_labels)
