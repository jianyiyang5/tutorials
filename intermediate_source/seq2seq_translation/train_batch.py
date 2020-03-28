import random
import os
import torch
from torch import nn, optim
from data import SOS_token, batch2TrainData, create_batches, prepareData
from model_batch import EncoderRNNBatch, LuongAttnDecoderRNN

# For pytorch 1.1
# def maskNLLLoss(inp, target, mask, device):
#     nTotal = (mask == True).sum()
#     crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
#     loss = crossEntropy.masked_select(mask == True).mean()
#     loss = loss.to(device)
#     return loss, nTotal.item()

def maskNLLLoss(inp, target, mask, device):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, clip, device, teacher_forcing_ratio=1.0):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.train()
    decoder.train()
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    # Perform backpropatation
    loss.backward()
    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()
    return sum(print_losses), n_totals


# iterations are actually epochs
def trainIters(model_name, src_voc, tgt_voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               encoder_n_layers, decoder_n_layers, hidden_size, save_dir, n_iteration, batch_size, clip, corpus_name,
               loadFilename, checkpoint, device):
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        batches = create_batches(pairs, batch_size)
        total_loss = 0
        n_totals = 0
        for batch in batches:
            input_variable, lengths, target_variable, mask, max_target_len = batch2TrainData(src_voc, tgt_voc, batch)
            # Run a training iteration with batch
            cur_batch_size = input_variable.size()[1]
            loss, n_total = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, cur_batch_size, clip, device)
            total_loss += loss
            n_totals += n_total

        # Print progress
        print_loss_avg = total_loss / n_totals
        print("Epoch: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))

        # Save checkpoint
        directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'iteration': iteration,
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': print_loss_avg,
            'src_voc_dict': src_voc.__dict__,
            'tgt_voc_dict': src_voc.__dict__,
            'src_embedding': encoder.embedding.state_dict(),
            'tgt_embedding': decoder.embedding.state_dict(),
            'attn_model': decoder.attn_model
        }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


if __name__ == '__main__':
    batch_size = 64
    clip = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    hidden_size = 256
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    attn_model = 'dot'
    checkpoint = None
    loadFilename = None
    epochs = 25

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True, '../data')

    encoder = EncoderRNNBatch(hidden_size, nn.Embedding(input_lang.n_words, hidden_size), encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, nn.Embedding(output_lang.n_words, hidden_size), hidden_size,
                                  output_lang.n_words, decoder_n_layers, dropout)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    trainIters('model_batch', input_lang, output_lang, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               encoder_n_layers, decoder_n_layers, hidden_size, 'output', epochs, batch_size, clip, 'en-fr',
               loadFilename, checkpoint, device)