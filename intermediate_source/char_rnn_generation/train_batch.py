import torch

from data import *
from utils import *
from model import *


def maskNLLLoss(inp, target, mask, device=None):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 2, target.unsqueeze(2)).squeeze(2))
    loss = crossEntropy.masked_select(mask).mean()
    if device:
        loss = loss.to(device)
    return loss, nTotal.item()


def train(rnn, category_lines, all_categories, batch_size=64, criterion=maskNLLLoss, learning_rate=0.005, epochs=600,
          print_every=10, plot_every=10):
    start = time.time()
    current_loss = 0
    all_losses = []
    for i in range(epochs):
        n_totals = 0
        batches = create_batches(category_lines, batch_size)
        j = 0
        # TODO no teacher forcing
        for batch in batches:
            rnn.zero_grad()
            inp, lengths, categories, target, mask, max_target_len = batch2TrainData(batch, all_categories)
            j += 1
            outputs, _ = rnn(inp, categories, lengths) #seq-len, batch, outsize
            loss, nTotal = criterion(outputs, target, mask)
            loss.backward()
            current_loss += loss.item()*nTotal
            n_totals += nTotal
            for p in rnn.parameters():
                # p.data.add_(-learning_rate, p.grad.data/lengths.size(0))
                p.data.add_(-learning_rate, p.grad.data)
        if i % print_every == 0:
            print(n_totals, current_loss)
            print('%d %d%% (%s) %.4f %s / %.10f' % (i, i / epochs * 100, timeSince(start), current_loss/n_totals, batch[0], learning_rate))
        if i % plot_every == 0:
            all_losses.append(current_loss/n_totals)
        current_loss = 0
        if len(all_losses) >= 2 and all_losses[-1] > all_losses[-2]:
            learning_rate *= 0.95
    return all_losses


def save_model(n_hidden_cat, n_hidden, n_categories, n_letters, model_dict, path):
    torch.save({'n_letters': n_letters,
                'n_categories': n_categories,
                'n_hidden_cat': n_hidden_cat,
                'n_hidden': n_hidden,
                'model_dict': model_dict}, path)
    print(f'Model is saved to {path}')


def load_model(path):
    checkpoint = torch.load(path)
    n_letters = checkpoint['n_letters']
    n_categories = checkpoint['n_categories']
    n_hidden_cat = checkpoint['n_hidden_cat']
    n_hidden = checkpoint['n_hidden']
    model_dict = checkpoint['model_dict']
    model = EncoderRNN(n_hidden_cat, n_hidden, torch.nn.Embedding(n_categories, n_hidden_cat),
                         torch.nn.Embedding(n_letters, n_hidden), n_letters)
    model.load_state_dict(model_dict)
    print(f'Load model from {path}')
    return model


if __name__ == '__main__':
    model_path = 'output/rnn_batch.pt'
    dn = os.path.dirname(model_path)
    os.makedirs(dn, exist_ok=True)
    n_hidden = 128
    n_hidden_cat = 16
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)
    rnn = EncoderRNN(n_hidden_cat, n_hidden, torch.nn.Embedding(n_categories, n_hidden_cat),
                         torch.nn.Embedding(n_letters, n_hidden), n_letters)
    save_model(n_hidden_cat, n_hidden, n_categories, n_letters, rnn.state_dict(), model_path)
    all_losses = train(rnn, category_lines, all_categories)
    # save_model(n_letters, n_hidden, n_categories, rnn.state_dict(), model_path)
    plot_losses(all_losses)
