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


def train(rnn, category_lines, all_categories, batch_size=64, criterion=maskNLLLoss, learning_rate=0.005, epochs=300,
          print_every=10, plot_every=10):
    start = time.time()
    current_loss = 0
    all_losses = []

    for i in range(epochs):
        batches = create_batches(category_lines, batch_size)
        j = 0
        for batch in batches:
            inp, lengths, categories, target, mask, max_target_len = batch2TrainData(batch, all_categories)
            j += 1
            outputs, _ = rnn(inp, lengths) #seq-len, batch, outsize
            outputs = outputs.transpose(0, 1)
            loss = criterion(outputs, target)
            loss.backward()
            current_loss += loss.item()
            for p in rnn.parameters():
                # p.data.add_(-learning_rate, p.grad.data/lengths.size(0))
                p.data.add_(-learning_rate, p.grad.data)
        if i % print_every == 0:
            print('%d %d%% (%s) %.4f %s / %.10f' % (i, i / epochs * 100, timeSince(start), loss, batch[0], learning_rate))
        if i % plot_every == 0:
            all_losses.append(current_loss/j)
        current_loss = 0
        # if len(all_losses) >= 2 and all_losses[-1] > all_losses[-2]:
        #     learning_rate *= 0.95
    return all_losses


if __name__ == '__main__':
    model_path = 'output/rnn_batch.pt'
    dn = os.path.dirname(model_path)
    os.makedirs(dn, exist_ok=True)
    n_hidden = 64
    n_hidden_cat = 8
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)
    rnn = EncoderRNN(n_hidden_cat, n_hidden, torch.nn.Embedding(n_categories, n_hidden_cat),
                         torch.nn.Embedding(n_letters, n_hidden), n_letters)
    all_losses = train(rnn, category_lines, all_categories)
    # save_model(n_letters, n_hidden, n_categories, rnn.state_dict(), model_path)
    plot_losses(all_losses)