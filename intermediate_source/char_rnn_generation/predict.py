
from train import *


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
    model = load_model('output/rnn.pt')
    model.eval()
    samples(model, 'Russian', all_categories, 'RUS')
    samples(model, 'German', all_categories, 'GER')
    samples(model, 'Spanish', all_categories, 'SPA')
    samples(model, 'Chinese', all_categories, 'CHINESE')
