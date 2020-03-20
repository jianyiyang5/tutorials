import torch.nn as nn
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


def train(rnn, category_lines, all_categories, batch_size=64, criterion=nn.NLLLoss(), learning_rate=0.005, epochs=300,
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
                # p.data.add_(-learning_rate, p.grad.data)
        guess, guess_i = categoryFromOutput(output[0], all_categories)
        correct = '✓' if guess == batch[0][1] else '✗ (%s)' % batch[0][1]
        print('%d %d%% (%s) %.4f %s / %s %s %.10f' % (i, i / epochs * 100, timeSince(start), loss, batch[0], guess, correct, learning_rate))
        all_losses.append(current_loss/j)
        current_loss = 0
        if len(all_losses) >= 2 and all_losses[-1] > all_losses[-2]:
            learning_rate *= 0.9
    return all_losses


# Just return an output given a line
def evaluate(rnn, line_tensor, lens):
    output, hidden = rnn(line_tensor, lens)
    return output


def save_model(input_size, hidden_size, output_size, model_dict, path):
    torch.save({'input_size': input_size,
                'hidden_size': hidden_size,
                'output_size': output_size,
                'model_dict': model_dict}, path)
    print(f'Model is saved to {path}')


def load_model(path):
    checkpoint = torch.load(path)
    n_hidden = checkpoint['hidden_size']
    n_categories = checkpoint['output_size']
    n_letters = checkpoint['input_size']
    model = EncoderRNN(n_hidden, nn.Embedding(n_letters, n_hidden), n_categories)
    model.load_state_dict(checkpoint['model_dict'])
    print(f'Load model from {path}')
    return model


if __name__ == '__main__':
    model_path = 'output/rnn_batch.pt'
    dn = os.path.dirname(model_path)
    os.makedirs(dn, exist_ok=True)
    n_hidden = 128
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)
    rnn = EncoderRNN(n_hidden, nn.Embedding(n_letters, n_hidden), n_categories)
    all_losses = train(rnn, category_lines, all_categories)
    save_model(n_letters, n_hidden, n_categories, rnn.state_dict(), model_path)
    plot_losses(all_losses)