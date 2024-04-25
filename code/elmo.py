import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext import vocab
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
from collections import Counter
import numpy as np

# Download NLTK tokenizer data
nltk.download('punkt')

# Determine device (GPU if available, otherwise CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())
    
    # List of prepositions to remove (customize as needed)
    prepositions = ['for', 'with', 'to', 'from', 'in', 'on', 'at', 'by']
    
    # Filter out prepositions and specific characters from words
    filtered_words = [word for word in words if word not in prepositions and word.isalnum()]
    
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
cleaned_df = clean_dataframe(df, text_column)

# Display the cleaned DataFrame
# print(cleaned_df.head())



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




def create_embedding_matrix(processed_data, glove_vectors):
    """
    Create an embedding matrix using pre-trained GloVe vectors for words in the vocabulary.

    Args:
    processed_data (object): Object containing processed data with a vocabulary.
    glove_vectors (object): Pre-trained GloVe vectors.

    Returns:
    embed_matrix (torch.Tensor): Embedding matrix where rows correspond to words in the vocabulary,
                                  initialized with GloVe vectors if available.
    """

    # Get vocabulary tokens
    vocab_tokens = processed_data.vocab.get_itos()
    
    # Determine embedding dimension
    embedding_dim = glove_vectors.vectors.shape[1]
    
    
    # Initialize embedding matrix
#     embed_matrix = torch.zeros(len(vocab_tokens), embedding_dim)
    embed_matrix = np.zeros((len(processed_data.vocab), embedding_dim))
    
    # Iterate over each word in the vocabulary
    for ind, word in enumerate(vocab_tokens):
        embedding_vector = None
        
        # Check if the word is in the GloVe vocabulary
        if word in glove_vectors.stoi:
            # Get the index of the word in the GloVe vocabulary
            glove_index = glove_vectors.stoi[word]
            
            # Get the embedding vector using the index
            embedding_vector = glove_vectors.vectors[glove_index]
        
        # Assign the embedding vector to the embedding matrix
        if embedding_vector is not None:
            embed_matrix[ind] = torch.tensor(embedding_vector)
    
    return embed_matrix


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
processed_data = ProcessYelp(cleaned_df, 5)
vocab_tokens = processed_data.vocab.get_itos()



label_train_dataset = LabelData(processed_data.vocab,cleaned_df,35)

index = 2  # Choose the index of the data instance you want to access
tokens, label = label_train_dataset[index]
tokens_list = tokens.tolist()  # Convert tensor to list
label_value = label.item()

BATCH_SIZE=64
embedding_dim=300
learning_rate=0.001
hidden_size=150
dropout=0.1
epochs=1

train_dataloader = DataLoader(label_train_dataset, batch_size=BATCH_SIZE, shuffle=False)

embedding_dict = {}

# file = open("./glove.6B/glove.6B.100d.txt", "r", encoding="utf-8")
glove_vectors = vocab.GloVe(name='6B', dim=embedding_dim)
embed_matrix=create_embedding_matrix(processed_data, glove_vectors)


embed_matrix_torch = torch.tensor(embed_matrix, dtype=torch.float64)
embed_matrix_torch = embed_matrix_torch.to(DEVICE)



elmo = (ELMo(len(processed_data.vocab), embedding_dim, hidden_size, dropout, embed_matrix_torch).double().to(DEVICE))
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(elmo.parameters(), lr=learning_rate)
print("Training start")

elmo.train()
val_prev = np.inf
for epoch in tqdm(range(epochs)):
    total_train_loss = 0
    total_loss = 0
    for sentence, label in (train_dataloader):
        inp = sentence[:, :-1].to(DEVICE)
        targ = sentence[:, 1:].to(DEVICE)
        optimizer.zero_grad()
        output = elmo(inp)
        loss = criterion(output, targ)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    print("loss:",total_train_loss / len(train_dataloader))




# for save the model

# output_path = 'bilstm.pt'
# torch.save(elmo.state_dict(), output_path)

print("Embedding created")




