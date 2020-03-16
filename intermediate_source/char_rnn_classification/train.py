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


# Just return an output given a line
def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


def create_confusion_matrix(rnn, category_lines, all_categories, n_confusion=10000):
    n_categories = len(all_categories)
    confusion = torch.zeros(n_categories, n_categories)
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        output = evaluate(rnn, line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    return confusion


if __name__ == '__main__':
    n_hidden = 128
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)
    rnn = RNN(n_letters, n_hidden, n_categories)
    all_losses = train(rnn, category_lines, all_categories)
    # print(all_losses)
    plot_losses(all_losses)
    confusion = create_confusion_matrix(rnn, category_lines, all_categories)
    plot_confusion_matrix(confusion, all_categories)