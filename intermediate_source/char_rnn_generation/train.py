import time

import torch.nn as nn

from data import *
from utils import *
from model import *


max_length = 20


def train_one_line(rnn, category_tensor, input_line_tensor, target_line_tensor, criterion=nn.NLLLoss(), learning_rate=0.0005):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item() / input_line_tensor.size(0)


def train(rnn, category_lines, all_categories, n_iters=100000, print_every=5000, plot_every=500):
    all_losses = []
    total_loss = 0  # Reset every plot_every iters
    start = time.time()
    for iter in range(1, n_iters + 1):
        output, loss = train_one_line(rnn, *randomTrainingExample(category_lines, all_categories))
        total_loss += loss
        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
    return all_losses


def save_model(input_size, n_categories, hidden_size, output_size, model_dict, path):
    torch.save({'input_size': input_size,
                'n_categories': n_categories,
                'hidden_size': hidden_size,
                'output_size': output_size,
                'model_dict': model_dict}, path)
    print(f'Model is saved to {path}')


def load_model(path):
    checkpoint = torch.load(path)
    model = RNN(checkpoint['input_size'], checkpoint['n_categories'], checkpoint['hidden_size'], checkpoint['output_size'])
    model.load_state_dict(checkpoint['model_dict'])
    print(f'Load model from {path}')
    return model


if __name__ == '__main__':
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)
    hidden_size = 128
    rnn = RNN(n_letters, n_categories, hidden_size, n_letters)
    all_losses = train(rnn, category_lines, all_categories, n_iters=100000)
    save_model(n_letters, n_categories, hidden_size, n_letters, rnn.state_dict(), 'output/rnn.pt')
    # plot_losses(all_losses)

