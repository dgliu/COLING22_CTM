import argparse
from utils.io import modify_original_data, get_law_index, get_accu_index, get_statistics_for_filter
from utils.io import filter_law_and_accu, filter_samples, split_seed_randomly


def main(args):
    # Set path
    save_path = args.path + args.problem
    train_path = args.path + args.problem + args.train
    valid_path = args.path + args.problem + args.valid
    test_path = args.path + args.problem + args.test

    if args.problem == 'big/':
        # If dataset is CAIL-big, we need to create a validation set
        split_seed_randomly(save_path, 'train.json', 'test.json', args.seed, args.ratio)

    if args.is_modify:
        # Modify the original data based on the findings of data exploration
        modify_original_data(train_path, valid_path, test_path, save_path)

        # Use new path
        train_path = args.path + args.problem + 'Modified_' + args.train
        valid_path = args.path + args.problem + 'Modified_' + args.valid
        test_path = args.path + args.problem + 'Modified_' + args.test

    # Count the frequency of accusations and laws, and filter them
    law_file = open(args.path + 'law.txt', 'r')
    law2num, num2law, total_law = get_law_index(law_file)

    accu_file = open(args.path + 'accu.txt', 'r', encoding='utf-8')
    accu2num, num2accu, total_accu = get_accu_index(accu_file)

    frequency_law, frequency_accu = get_statistics_for_filter(train_path, valid_path, total_law, law2num,
                                                              total_accu, accu2num)

    filter_law_list, filter_law2num, filter_accu_list, filter_accu2num = filter_law_and_accu(save_path, total_law,
                                                                                             num2law, frequency_law,
                                                                                             total_accu, num2accu,
                                                                                             frequency_accu)

    # Filter the samples according to the filtered accusations and laws set
    filter_samples(save_path, train_path, valid_path, test_path, law2num, filter_law_list,
                   filter_law2num, accu2num, filter_accu_list, filter_accu2num)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('-p', dest='path', default='datax/')
    parser.add_argument('-d', dest='problem', default='small/')  # or 'big/'
    parser.add_argument('-i', dest='is_modify', action='store_true', default=False)
    parser.add_argument('-tr', dest='train', help='train set', default='data_train.json')
    parser.add_argument('-v', dest='valid', help='valid set', default='data_valid.json')
    parser.add_argument('-te', dest='test', help='test set', default='data_test.json')
    parser.add_argument('-s', dest='seed', help='random seed', type=int, default=0)
    parser.add_argument('-r', dest='ratio', help='ratio for validation', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
