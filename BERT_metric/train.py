### NAME:           train.py
### PURPOSE:        Training of the BERT metric.
### AUTHOR:         Mortiz Altemeyer, Philipp Hege
### LAST CHANGE:    24.08.2022

# Import packages
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datetime import datetime
from os import makedirs
from sklearn.metrics import mean_squared_error
from tools import LoadAndPreprocessAndNormalizeData # local .py file


def train_metric(DATA_PATH = 'data/data_2017', # for Windows: 'data\data_2017'
                 MODEL_NAME = 'bert-base-multilingual-uncased',
                 LEARNING_RATE = 2e-5,
                 PER_DEVICE_TRAIN_BATCH_SIZE = 16,
                 PER_DEVICE_TEST_BATCH_SIZE = 16,
                 NUM_TRAIN_EPOCHS = 5,
                 WEIGHT_DECAY = 0.01,
                 NUMBER_DATAPOINTS = 500,
                 TEST_SIZE = 0.2):
    '''
    Function that generates the model and trains it as our standard BERT metric.
    
    :param DATA_PATH: String. Default: 'data/data_2017'. Path of training and test data (used within training).
    :param MODEL_NAME: String. Default: 'bert-base-multilingual-uncased'. Name of the model used from HuggingFace.
    :param LEARNING_RATE: Float > 0. Defines the learning speed within the gradient step of the optimization.
    :param PER_DEVICE_TRAIN_BATCH_SIZE: Integer. Default: 16. Batch size per device/core in training.
    :param PER_DEVICE_TEST_BATCH_SIZE: Integer. Default: 16. Batch size per device/core in testing.
    :param NUM_TRAIN_EPOCHS: Integer. Default: 5. Number of training epochs.
    :param WEIGHT_DECAY: Float. Weight decay during training step.
    :param NUMBER_DATAPOINTS: Integer > 0. Default: 500. Number of data points used for training and testing (within training).
    :param TEST_SIZE: Float \element (0, 1). Default: 0.2. Defines the percentage of test data from the whole data (=NUMBER_DATAPOINTS). E.g. NUMBER_DATAPOINTS=1_000, TEST_SIZE=0.3 -> 700/300 train/test.
    
    :return trainer: Trainer object. Resulting trainer after training process.
    :return runtime: Float. Runtime of the training process measured in seconds.
    '''
    # Extract tokens for scrs and hyps
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create data_collator object
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 1, problem_type = 'regression')
    
    # Create dataset objects
    ds_train, ds_test = LoadAndPreprocessAndNormalizeData(DATA_PATH + '_train.csv', tokenizer, NUMBER_DATAPOINTS, TEST_SIZE)
    
    def compute_metrics_for_regression(EVAL_PRED):
        '''
        Function that computes MSE as regression objective. Used within RegressionTrainer.
        
        :param EVAL_PRED: Touple. Contains two lists, one with predictions, the other with the labels/scores to evaluate.
        
        :return:
        Dict. Contains one key, MSE as value.
        '''
        predictions, labels = EVAL_PRED
        mse = mean_squared_error(labels, predictions, squared=False)
        return {'mse': mse}
    
    # Definde training arguments
    training_args = TrainingArguments(
        output_dir = './results',
        learning_rate = LEARNING_RATE,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = PER_DEVICE_TEST_BATCH_SIZE,
        num_train_epochs = NUM_TRAIN_EPOCHS,
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        save_total_limit = 2,
        metric_for_best_model = 'mse',
        load_best_model_at_end = True,
        weight_decay = WEIGHT_DECAY)
    
    class RegressionTrainer(Trainer):
        '''
        Class that defines the Regression trainer. Child/Subclass of the Trainer class from the transformers package.
        '''
        def compute_loss(self,
                         model,
                         inputs,
                         return_outputs = False):
            '''
            Method to compute the loss of a model given its inputs.
            
            :param model: Model object. Model to compute the loss for.
            :param inputs: Dataset object. Inputs for the model to compute the loss for.
            :param return_outputs. Bool. Default: False. Whether or not outputs of the model should be returned.
            
            :return:
            Float. Loss of the model. If RETURN_OUTPUTS=True: Tuple. First element loss (Float), second element outputs of the model (list).
            '''
            # get labels, unpack the and predict
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            # compute logits and loss
            logits = outputs[0][:, 0]
            loss = torch.nn.functional.mse_loss(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    # Create trainer object
    trainer = RegressionTrainer(
        model = model,
        args = training_args,
        train_dataset = ds_train,
        eval_dataset = ds_test,
        compute_metrics = compute_metrics_for_regression)
    
    t0 = datetime.now() # measure time: start
    
    # Fine-tune the mode
    trainer.train()
    
    t1 = datetime.now() # measure time: end
    runtime = (t1 - t0).total_seconds() # get runtime in seconds
    
    # Save model and tokenizer (in this case: ./ creates directory if not existing)
    model_save_directory = f'./model_checkpoints/{MODEL_NAME}__{NUM_TRAIN_EPOCHS}__{NUMBER_DATAPOINTS}__{TEST_SIZE}'
    tokenizer.save_pretrained(model_save_directory)
    model.save_pretrained(model_save_directory)
    
    return trainer, runtime


# Define train_metric_adapter function
def train_metric_adapter(DATA_PATH = 'data/data_2017', # for Windows: 'data\data_2017'
                         MODEL_NAME = 'bert-base-multilingual-uncased',
                         LEARNING_RATE = 2e-5,
                         PER_DEVICE_TRAIN_BATCH_SIZE = 16,
                         PER_DEVICE_TEST_BATCH_SIZE = 16,
                         NUM_TRAIN_EPOCHS = 5,
                         WEIGHT_DECAY = 0.01,
                         NUMBER_DATAPOINTS = 500,
                         TEST_SIZE = 0.2):
    '''
    Function that generates the model and trains it as our BERT metric with adapters.
    
    :param DATA_PATH: String. Default: 'data/data_2017'. Path of training and test data (used within training).
    :param MODEL_NAME: String. Default: 'bert-base-multilingual-uncased'. Name of the model used from HuggingFace.
    :param LEARNING_RATE: Float > 0. Defines the learning speed within the gradient step of the optimization.
    :param PER_DEVICE_TRAIN_BATCH_SIZE: Integer. Default: 16. Batch size per device/core in training.
    :param PER_DEVICE_TEST_BATCH_SIZE: Integer. Default: 16. Batch size per device/core in testing.
    :param NUM_TRAIN_EPOCHS: Integer. Default: 5. Number of training epochs.
    :param WEIGHT_DECAY: Float. Weight decay during training step.
    :param NUMBER_DATAPOINTS: Integer > 0. Default: 500. Number of data points used for training and testing (within training).
    :param TEST_SIZE: Float \element (0, 1). Default: 0.2. Defines the percentage of test data from the whole data (=NUMBER_DATAPOINTS). E.g. NUMBER_DATAPOINTS=1_000, TEST_SIZE=0.3 -> 700/300 train/test.
    
    :return trainer: Trainer object. Resulting trainer after training process.
    :return runtime: Float. Runtime of the training process measured in seconds.
    '''
    # Extract tokens for scrs and hyps
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create data_collator object
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 1, problem_type = 'regression')
    model.add_adapter('Regression')
    model.train_adapter('Regression')
    
    # Create dataset objects
    ds_train, ds_test = LoadAndPreprocessAndNormalizeData(DATA_PATH + '_train.csv', tokenizer, NUMBER_DATAPOINTS, TEST_SIZE)
    
    def compute_metrics_for_regression(EVAL_PRED):
        '''
        Function that computes MSE as regression objective. Used within RegressionTrainer.
        
        :param EVAL_PRED: Touple. Contains two lists, one with predictions, the other with the labels/scores to evaluate.
        
        :return:
        Dict. Contains one key, MSE as value.
        '''
        predictions, labels = EVAL_PRED
        mse = mean_squared_error(labels, predictions, squared=False)
        return {'mse': mse}
    
    # Definde training arguments
    training_args = TrainingArguments(
        output_dir = './results',
        learning_rate = LEARNING_RATE,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = PER_DEVICE_TEST_BATCH_SIZE,
        num_train_epochs = NUM_TRAIN_EPOCHS,
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        save_total_limit = 2,
        metric_for_best_model = 'mse',
        load_best_model_at_end = True,
        weight_decay = WEIGHT_DECAY)
    
    class RegressionTrainer(Trainer):
        '''
        Class that defines the Regression trainer. Child/Subclass of the Trainer class from the transformers package.
        '''
        def compute_loss(self,
                         model,
                         inputs,
                         return_outputs = False):
            '''
            Method to compute the loss of a model given its inputs.
            
            :param model: Model object. Model to compute the loss for.
            :param inputs: Dataset object. Inputs for the model to compute the loss for.
            :param return_outputs. Bool. Default: False. Whether or not outputs of the model should be returned.
            
            :return:
            Float. Loss of the model. If RETURN_OUTPUTS=True: Tuple. First element loss (Float), second element outputs of the model (list).
            '''
            # get labels, unpack the and predict
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            # compute logits and loss
            logits = outputs[0][:, 0]
            loss = torch.nn.functional.mse_loss(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    # Create trainer object
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        compute_metrics = compute_metrics_for_regression)
    
    t0 = datetime.now() # measure time: start
    
    # Fine-tune the mode
    trainer.train()
    
    t1 = datetime.now() # measure time: end
    runtime = (t1 - t0).total_seconds() # get runtime in seconds
    
    # Save model, tokenizer and all adapters (in this case: ./ creates directory if not existing)
    model_save_directory = f'./model_checkpoints/{MODEL_NAME}_adapter__{NUM_TRAIN_EPOCHS}__{NUMBER_DATAPOINTS}__{TEST_SIZE}'
    tokenizer.save_pretrained(model_save_directory)
    model.save_all_adapters(model_save_directory)
    model.save_pretrained(model_save_directory)
    
    return trainer, runtime