import os
import numpy as np
import tensorflow as tf
import pickle
from utils import log, data_path

np.random.seed(1)
tf.set_random_seed(1)

data_file_path = data_path + os.sep + "books_in_sentences"

MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 10
illegal_seqs = ["isbn"]

TOKEN_TAB = 0
TOKEN_BOS = 27
TOKEN_EOS = 28


def element2token(elem):
    return ord(elem)-96


def is_valid_seq(seq):
    """Filter what words to add to the data set."""
    valid = True
    seq_len = len(seq)
    
    # Only accept a certain length
    if seq_len > MAX_SEQ_LEN or seq_len < MIN_SEQ_LEN:
        valid = False
    
    # Checks if the word only condtains letters
    if not seq.isalpha():
        valid = False
        
    # Check a hand-crafted list of illegal words
    for illegal_word in illegal_seqs:
        if illegal_word == seq:
            valid = False
    
    return valid


def parse_lines(nr_lines):
    """Reads a number of lines from the input file, and returns a set of all valid words in those lines."""
    set_of_seqs = set()
    with open(data_file_path) as file:
        for line in file:
            seqs = line.split()

            for seq in seqs:
                seq = seq.lower()  # Turns word to lower case
                if is_valid_seq(seq):
                    set_of_seqs.add(seq)

            nr_lines -= 1
            if nr_lines == 0:
                break

    return set_of_seqs


def tokenize_seqs(seqs):
    """Returns a list of arrays of tokenized letters. Adds start and end symbol."""
    list_of_tokenized_seqs = []
    for word in seqs:
        tokenized_seq = [TOKEN_BOS]
        for character in word:
            tokenized_seq.append(element2token(character))
        tokenized_seq.append(TOKEN_EOS)

        list_of_tokenized_seqs.append(tokenized_seq)
    return list_of_tokenized_seqs


def load_data_from_source(nr_lines_to_parse=1000000, save=True):
    log("Scanning the first", nr_lines_to_parse, "lines.")

    list_of_seqs = parse_lines(nr_lines_to_parse)
    nr_seqs = len(list_of_seqs)

    log("Number of words found:", nr_seqs)
    log("Preprocessing words.")

    list_of_tokenized_seqs = tokenize_seqs(list_of_seqs)

    log("Shuffling and splitting data.")

    train_set_size = int(nr_seqs*0.75)
    validation_set_size = int(train_set_size*0.25)
    train_set_size = train_set_size-validation_set_size
    test_set_size = nr_seqs-(train_set_size+validation_set_size)

    train_set = list_of_tokenized_seqs[:train_set_size]
    validation_set = list_of_tokenized_seqs[train_set_size:train_set_size+validation_set_size]
    test_set = list_of_tokenized_seqs[train_set_size+validation_set_size:]
    
    vocab = [chr(i+97) for i in range(0, 26)]+["S", "E"]

    char2int = {chr(i+97): i for i in range(0, 26)}
    char2int['S'] = TOKEN_BOS
    char2int['E'] = TOKEN_EOS
    char2int['T'] = TOKEN_TAB
    
    if save:
        save_data(train_set, validation_set, test_set, vocab, char2int)
        
    return train_set, validation_set, test_set, vocab, char2int


def load_data_from_saves():
    log("Loading data from saved files.")

    with open(data_path + os.sep + "words_train.dat", "rb") as inFile:
        train_set = pickle.load(inFile)
    with open(data_path + os.sep + "words_validation.dat", "rb") as inFile:
        validation_set = pickle.load(inFile)
    with open(data_path + os.sep + "words_test.dat", "rb") as inFile:
        test_set = pickle.load(inFile)
    with open(data_path + os.sep + "vocab.dat", "rb") as inFile:
        vocab = pickle.load(inFile)
    with open(data_path + os.sep + "char2int.dict", "rb") as inFile:
        char2int = pickle.load(inFile)

    log("Done loading data.")

    return train_set, validation_set, test_set, vocab, char2int


def save_data(train_set, validation_set, test_set, vocab, char2int):
    log("Saving words.")

    # Save the tokenized words
    with open(data_path + os.sep + "words_train.dat", "wb") as outFile:
        pickle.dump(train_set, outFile)
    with open(data_path + os.sep + "words_validation.dat", "wb") as outFile:
        pickle.dump(validation_set, outFile)
    with open(data_path + os.sep + "words_test.dat", "wb") as outFile:
        pickle.dump(test_set, outFile)

    # Save the vocabulary
    with open(data_path + os.sep + "vocab.dat", "wb") as outFile:
        pickle.dump(vocab, outFile)

    # Save the mapping
    with open(data_path + os.sep + "char2int.dict", "wb") as outFile:
        pickle.dump(char2int, outFile)

    log("Done saving words.")