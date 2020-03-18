#TODO batch train
#x2 = x.transpose(0, 1)
#y = x2[torch.arange(x2.size(0)), idx]
import torch.nn as nn
from torch.nn.modules.rnn import RNN
from data import *
from utils import *
from model import EncoderRNN


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


def train(rnn, category_lines, all_categories, batch_size=64, criterion=nn.NLLLoss(), learning_rate=0.005, epochs=20,
          print_every=5000, plot_every=1000):
    start = time.time()
    current_loss = 0
    all_losses = []

    for i in range(epochs):
        batches = create_batches(category_lines, batch_size)
        j = 0
        for batch in batches:
            inp, lengths, target = batch2TrainData(batch, all_categories)
            j += 1
            outputs, _ = rnn(inp, lengths) #seq-len, batch, outsize
            outputs = outputs.transpose(0, 1)
            output = outputs[torch.arange(outputs.size(0)), lengths-1]
            loss = criterion(output, target)
            loss.backward()
            current_loss += loss.item()
            for p in rnn.parameters():
                p.data.add_(-learning_rate, p.grad.data/lengths.size(0))
        print(f'epoch {i} {round(current_loss/j, 4)}')
        all_losses.append(current_loss/j)
        current_loss = 0
            # Print iter number, loss, name and guess
            # if iter % print_every == 0:
            #     guess, guess_i = categoryFromOutput(outputs[0], all_categories)
            #     correct = '✓' if guess == batches[0][1] else '✗ (%s)' % batches[0][1]
            #     print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
            # # Add current loss avg to list of losses
            # if iter % plot_every == 0:
            #     all_losses.append(current_loss / plot_every)
            #     current_loss = 0
    return all_losses


# Just return an output given a line
def evaluate(rnn, line_tensor):
    #hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


def create_confusion_matrix(rnn, category_lines, all_categories, n_confusion=10000):
    n_categories = len(all_categories)
    confusion = torch.zeros(n_categories, n_categories)
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        output = evaluate(rnn, line_tensor)
        guess, guess_i = categoryFromOutput(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    return confusion


def save_model(input_size, hidden_size, output_size, model_dict, path):
    torch.save({'input_size': input_size,
                'hidden_size': hidden_size,
                'output_size': output_size,
                'model_dict': model_dict}, path)
    print(f'Model is saved to {path}')


def load_model(path):
    checkpoint = torch.load(path)
    model = RNN(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['output_size'])
    model.load_state_dict(checkpoint['model_dict'])
    print(f'Load model from {path}')
    return model


if __name__ == '__main__':
    model_path = 'output/rnn_batch.pt'
    dn = os.path.dirname(model_path)
    os.makedirs(dn, exist_ok=True)
    n_hidden = 8
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)
    rnn = EncoderRNN(n_hidden, nn.Embedding(n_letters, n_hidden), n_categories)
    all_losses = train(rnn, category_lines, all_categories)
    save_model(n_letters, n_hidden, n_categories, rnn.state_dict(), model_path)
    # print(all_losses)
    # plot_losses(all_losses)
    # confusion = create_confusion_matrix(rnn, category_lines, all_categories)
    # plot_confusion_matrix(confusion, all_categories)