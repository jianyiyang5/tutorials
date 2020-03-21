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


# Sample from a category and starting letter
def sample(rnn, category, all_categories, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category, all_categories)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(rnn, category, all_categories, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(rnn, category, all_categories, start_letter))


if __name__ == '__main__':
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)
    hidden_size = 128
    rnn = RNN(n_letters, n_categories, hidden_size, n_letters)
    all_losses = train(rnn, category_lines, all_categories, n_iters=100000)
    # plot_losses(all_losses)

    samples(rnn, 'Russian',all_categories, 'RUS')
    samples(rnn, 'German', all_categories,'GER')
    samples(rnn, 'Spanish',all_categories, 'SPA')
    samples(rnn, 'Chinese',all_categories, 'CHI')
