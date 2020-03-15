from model import *
from data import *
from utils import *


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def train_one_line(category_tensor, line_tensor, rnn, criterion=nn.NLLLoss(), learning_rate=0.005):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()


def train(rnn, category_lines, all_categories, criterion=nn.NLLLoss(), learning_rate=0.005, n_iters=100000,
          print_every=5000, plot_every=1000):
    start = time.time()
    current_loss = 0
    all_losses = []

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        output, loss = train_one_line(category_tensor, line_tensor, rnn, criterion=criterion, learning_rate=learning_rate)
        current_loss += loss
        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    return all_losses


if __name__ == '__main__':
    n_hidden = 128
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)
    rnn = RNN(n_letters, n_hidden, n_categories)
    all_losses = train(rnn, category_lines, all_categories)
    print(all_losses)