
from train_batch import *


# Sample from a category and starting letter
def sample(rnn, category, all_categories, start_letter='A', max_length=20):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryIdxTensor([category], all_categories)
        input, lengths = inputVar([start_letter])

        output_name = start_letter
        hidden = None

        for i in range(max_length):
            outputs, hidden = rnn(input, category_tensor, lengths, hidden)
            output = outputs.transpose(0, 1)[0]
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input, lengths = inputVar([letter])

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(rnn, category, all_categories, start_letters='ABC'):
    for start_letter in start_letters:
        print(category, start_letter, sample(rnn, category, all_categories, start_letter))


if __name__ == '__main__':
    category_lines, all_categories = load_data()
    model = load_model('output/rnn_batch.pt')
    model.eval()
    for category in ['Russian', 'German', 'Spanish', 'Chinese', 'Korean', 'Japanese', 'French', 'Italian', 'English']:
        samples(model, category, all_categories, string.ascii_uppercase)
        print()

    category = 'Chinese'
    start_letters = 'An'
    print(category, start_letters, sample(model, category, all_categories, start_letters))
