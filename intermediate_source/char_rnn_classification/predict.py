from train import *


def predict(model, all_categories, input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(model, lineToTensor(input_line))
        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


if __name__ == '__main__':
    model = load_model('output/rnn.pt')
    model.eval()
    category_lines, all_categories = load_data()
    names = ['Dovesky', 'Jackson', 'Satoshi', 'Chan', 'Jonas', 'Joris']
    for name in names:
        predict(model, all_categories, name)