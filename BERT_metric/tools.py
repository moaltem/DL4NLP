### NAME:           tools.py
### PURPOSE:        Assisting functions for basic processes used in BERT metric.
### AUTHOR:         Mortiz Altemeyer, Philipp Hege
### LAST CHANGE:    24.08.2022

# Import packages
import os
import numpy as np
from datasets import load_dataset, Dataset


def WriteOutput_txt(INPUT,
                    PATH,
                    FILENAME):
    '''
    Function that writes list data in a .txt-file line by line. Each element of the input list will be written in a new line.
    
    :param INPUT: List. Input that should be written.
    :param PATH: String. Directory of the output file.
    :param FILENAME: String. Name of the output file.
    
    :return:
    None. But: Writes the output file as .txt in the specified directory with specified filename.
    '''
    try:
        # create directory of it does not exist
        isExist = os.path.exists(PATH)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(PATH)
            print(f'New directory created: {PATH}')
    
        # write input line by line
        with open(os.path.join(PATH, FILENAME), 'w') as f:
            for line in INPUT:
                f.write(line)
                f.write('\n')
        print('File written successfully: ', os.path.join(PATH, FILENAME))
    except Exception as E:
        print(E)
            
            
def StandardDeviation(PREDICTIONS,
                      LABELS):
    '''
    Function that computes the standard deviation of some predictions and their corresponding true labels/scores.
    
    :param PREDICTIONS: List/np.array. Predictions of a model.
    :param LABELS: List/np.array. True labels/scores of the model, corresponding to the given predictions.
    
    :return:
    Float. Computed standard deviation.
    '''
    res = 0
    # for each element in the list
    for ii in range(len(PREDICTIONS)):
        # extract predictions and labels
        pred = PREDICTIONS[ii]
        lab = LABELS[ii]
        # compute standard deviation
        res += np.sqrt((pred - lab)**2)
    return res/len(PREDICTIONS)


def NormalizeData(DATA):
    '''
    Function to normalize the labels/scores of a Dataset object. Yields a label/score ranging from 0 to 1. Normalization(x) = (x - x_min) / (x_max - x_min).
    
    :param DATA: Dataset object (datasets.arrow_dataset.Dataset). Data which should be normalized.
    
    :return:
    Dict. Normalized data as a dictionary with one key: Labels. Compatible to integrate it into the original Dataset object later on.
    '''
    return {'labels': (np.array(DATA['labels']) - np.min(np.array(DATA['labels']))) / (np.max(np.array(DATA['labels'])) - np.min(np.array(DATA['labels'])))}


def LoadAndPreprocessAndNormalizeData(DATA_FILES,
                                      TOKENIZER,
                                      DATA_POINTS_LOADED = None,
                                      TEST_SIZE = None, # between 0 and 1
                                      SEED=42):
    '''
    Function that loads, pre-processes and normalizes a Dataset object. For normalization see function above. Pre-processing consists of tokenization of references and hypotheses.
    
    :param DATA_FILES: String. Directory, filename and extension of the data to be loaded.
    :param TOKENIZER: Tokenizer object. Tokenizer that is used for tokenization within pre-processing.
    :param DATA_POINTS_LOADED: Integer. Default: None. Number of data ponts that should be loaded. When default, all will be loaded.
    :param TEST_SIZE: Float \element (0, 1). Default: None. Defines the percentage of test data of the train and test split. When default, no split is done.
    :param SEED: Integer. Defines the seed to make the results reproducable. Default: 42.
    
    :return:
    Dataset object. Final data after loading, pre-processing and normalization.
    '''
    # locally define function for pre-processing
    def PreprocessData(DATA):
        '''
        Function applying a tokenizer to the references (refs) as well as the hypotheses (hyps) of an input Dataset object.
        
        :param DATA: Dataset object (datasets.arrow_dataset.Dataset). Data which should be pre-processed.
        
        :return:
        Dataset objects. Data (refs and hyps) after tokenization. One data set if no split is done, two (train and test) if a split is done.
        '''
        return TOKENIZER(DATA['refs'], DATA['hyps'], truncation = True, padding = 'max_length', max_length = 256)
    
    # load dataset
    if TEST_SIZE is not None:
        # load, shuffle (ensure that we have a mixed data set when slicing) and then do the split
        data = load_dataset('csv', data_files = DATA_FILES, split = 'train').shuffle(seed=SEED).train_test_split(test_size=TEST_SIZE)
        
        # check if slizing is possible
        assert (data['train'].num_rows >= int((1-TEST_SIZE)*DATA_POINTS_LOADED)) and (data['test'].num_rows >= int(TEST_SIZE*DATA_POINTS_LOADED)), 'Either DATA_POINTS_LOADED or TEST_SIZE is too large for the given data!'
        
        # slice train and test data if not the whole data set should be loaded
        train_data = Dataset.from_dict(data['train'][:int((1-TEST_SIZE)*DATA_POINTS_LOADED)])
        test_data = Dataset.from_dict(data['test'][:int(TEST_SIZE*DATA_POINTS_LOADED)])
        del data # delete big data variable
        
        # normalize the labels of the train and test data
        train_data = train_data.map(NormalizeData, batched = True)
        test_data = test_data.map(NormalizeData, batched = True)
        
        # use preprocessing function with mapping
        train_data = train_data.map(PreprocessData, batched = True)
        test_data = test_data.map(PreprocessData, batched = True)
        
        # return train and test dataset object
        return train_data, test_data
    
    else:
        # load and shuffle (ensure that we have a mixed data set when slicing)
        data = load_dataset('csv', data_files = DATA_FILES, split = 'train').shuffle(seed=SEED)
        
        # check if slizing is possible
        assert data.num_rows >= DATA_POINTS_LOADED, 'DATA_POINTS_LOADED is too large for the given data!'
        
        # slice data if not the whole data set should be loaded
        data = Dataset.from_dict(data[:DATA_POINTS_LOADED])
        
        # normalize labels of the data
        data = data.map(NormalizeData, batched = True)
        
        # use preprocessing function with mapping
        data = data.map(PreprocessData, batched = True)
        
        # return dataset object
        return data