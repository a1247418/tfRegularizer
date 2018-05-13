import os
import numpy as np
import tensorflow as tf
import pickle
np.random.seed(1)
tf.set_random_seed(1)

PRINT_OUT = True

path = os.getcwd()
dataPath = path + os.sep + "data"
dataFilePath = dataPath + os.sep + "books_in_sentences"

MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 10
illegalSeqs = ["isbn"]

TOKEN_TAB = 0
TOKEN_BOS = 27   # TODO: is this necessary?
TOKEN_EOS = 28

def element2token(elem):
    return ord(elem)-96


def isValidSeq(seq):
    """Filter what words to add to the data set."""
    valid = True
    seqLen = len(seq)
    
    #Only accept a certain length
    if(seqLen > MAX_SEQ_LEN or seqLen < MIN_SEQ_LEN):
        valid = False
    
    #Checks if the word only condtains letters
    if( not seq.isalpha() ):
        valid = False
        
    #Check a hand-crafted list of illegal words
    for illegalWord in illegalSeqs:
        if( illegalWord == seq ):
            valid = False
    
    return valid


def parseLines(nrLines):
    """Reads a number of lines from the input file, and returns a set of all valid words in those lines."""
    setOfSeqs = set()
    with open(dataFilePath) as file:
        for line in file:
            seqs = line.split()

            for seq in seqs:
                seq = seq.lower()#Turns word to lower case
                if( isValidSeq(seq) ):
                    setOfSeqs.add(seq)

            nrLines -= 1
            if(nrLines == 0):
                break

    return setOfSeqs


def tokenizeSeqs(seqs):
    """Returns a list of arrays of tokenized letters. Adds start and end symbol."""
    listOfTokenizedSeqs = []
    for word in seqs:
        tokenizedSeq = [TOKEN_BOS]
        for character in word:
            tokenizedSeq.append(element2token(character))#makes letters numbers, with 'a' = 0
        tokenizedSeq.append(TOKEN_EOS)

        listOfTokenizedSeqs.append(tokenizedSeq)
    return listOfTokenizedSeqs


def loadDataFromSource(nrLinesToParse = 1000000, save = True):
    if PRINT_OUT: print("Scanning the first",nrLinesToParse,"lines.")

    listOfSeqs = parseLines(nrLinesToParse)
    nrSeqs = len(listOfSeqs)

    if PRINT_OUT: print("Number of words found:",nrSeqs)
    if PRINT_OUT: print("Preprocessing words.")

    listOfTokenizedSeqs = tokenizeSeqs(listOfSeqs)

    if PRINT_OUT: print("Shuffling and splitting data.")

    trainSetSize = int(nrSeqs*0.75)
    validationSetSize = int(trainSetSize*0.25)
    trainSetSize = trainSetSize-validationSetSize
    testSetSize = nrSeqs-(trainSetSize+validationSetSize)

    trainSet = listOfTokenizedSeqs[:trainSetSize]
    validationSet = listOfTokenizedSeqs[trainSetSize:trainSetSize+validationSetSize]
    testSet = listOfTokenizedSeqs[trainSetSize+validationSetSize:]
    
    vocab = [chr(i+97) for i in range(0,26)]+["S","E"]    

    char2int = {chr(i+97):i for i in range(0,26)}
    char2int['S'] = -1
    char2int['E'] = -2
    char2int['T'] = -3
    
    if(save):
        saveData(trainSet, validationSet, testSet, vocab, char2int)
        
    return trainSet, validationSet, testSet, vocab, char2int


def loadDataFromSaves():
    trainSet = []
    validationSet = []
    testSet = []
    vocab = []
    char2int = {}

    if PRINT_OUT: print("Loading data from saved files.")

    with open(dataPath + os.sep + "words_train.dat","rb") as inFile:
        trainSet = pickle.load(inFile)
    with open(dataPath + os.sep + "words_validation.dat","rb") as inFile:
        validationSet = pickle.load(inFile)
    with open(dataPath + os.sep + "words_test.dat","rb") as inFile:
        testSet = pickle.load(inFile)
    with open(dataPath + os.sep + "vocab.dat","rb") as inFile:
        vocab = pickle.load(inFile)
    with open(dataPath + os.sep + "char2int.dict","rb") as inFile:
        char2int = pickle.load(inFile)

    if PRINT_OUT: print("Done loading data.")

    return trainSet, validationSet, testSet, vocab, char2int


def saveData(trainSet, validationSet, testSet, vocab, char2int):
    if PRINT_OUT: print("Saving words.")

    #Save the tokenized words
    with open(dataPath + os.sep + "words_train.dat","wb") as outFile:
        pickle.dump(trainSet, outFile)
    with open(dataPath + os.sep + "words_validation.dat","wb") as outFile:
        pickle.dump(validationSet, outFile)
    with open(dataPath + os.sep + "words_test.dat","wb") as outFile:
        pickle.dump(testSet, outFile)

    #Save the vocabulary
    with open(dataPath + os.sep + "vocab.dat","wb") as outFile:
        pickle.dump(vocab, outFile)

    #Save the mapping
    with open(dataPath + os.sep + "char2int.dict","wb") as outFile:
        pickle.dump(char2int, outFile)

    if PRINT_OUT: print("Done saving words.")