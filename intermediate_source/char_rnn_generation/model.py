import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, n_categories, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class EncoderRNN(nn.Module):
    def __init__(self, cat_hidden_size, hidden_size, category_embedding, embedding, output_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        # self.n_categories = n_categories
        # self.hidden_size = hidden_size
        self.category_embedding = category_embedding
        self.embedding = embedding
        self.output_size = output_size

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(cat_hidden_size+hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, cat_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        category_embedded = self.category_embedding(cat_seq)
        # repeat with the max seq length
        category_embedded = category_embedded.repeat(max(input_lengths).item(), 1, 1)
        combined = torch.cat((embedded, category_embedded), 2)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(combined, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        outputs = self.linear(outputs)
        outputs = self.softmax(outputs)
        return outputs, hidden