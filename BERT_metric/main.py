### NAME:           main.py
### PURPOSE:        Execution of training and evaluation of the BERT metric.
### AUTHOR:         Mortiz Altemeyer, Philipp Hege
### LAST CHANGE:    24.08.2022

# Import packages
from train import train_metric, train_metric_adapter # local .py file
from evaluate import evaluate_metric # local .py file
from tools import WriteOutput_txt # local .py file


if __name__ == '__main__':
    # global variables
    test_size = 0.2 # percentage used to test within training
    iterations = 5 # number of iteratins
    model_name = 'bert-base-multilingual-uncased' # model name (from HuggingFace)
    epochs__nbs_datapoints = [(1, 500), (8, 500), (1, 5_000), (8, 5_000)] # First element of the tuple: number of epochs per training iteration; Second: number of data points used for each training iteration
    eval_size = 1_000 # number of data points for evaluation
    
    # make training and evaluation for each iteration and write averaged results
    for epoch, nb_datapoints in epochs__nbs_datapoints:
        path = f'./results/{model_name}__{epoch}__{nb_datapoints}__{test_size}/'
        
        # STANDARD model
        # measure average runtime training, runtime evaluation, correlation, standard deviation and mse
        avg_rt_train, avg_rt_eval, avg_corr, avg_std, avg_mse = 0, 0, 0, 0, 0
        for _ in range(iterations):
            # training
            trainer, rt_train = train_metric(NUM_TRAIN_EPOCHS = epoch,
                                             NUMBER_DATAPOINTS = nb_datapoints,
                                             TEST_SIZE = test_size,
                                             MODEL_NAME = model_name)
            # evaluation
            rt_eval, corr, _, std, mse = evaluate_metric(trainer,
                                                         (epoch, nb_datapoints, test_size),
                                                         EVAL_SIZE = eval_size,
                                                         SCATTER_PATH = path)
            # count up measures, return averages later (divided by the number of iterations)
            avg_rt_train += rt_train
            avg_rt_eval += rt_eval
            avg_corr += corr
            avg_std += std
            avg_mse += mse
        # write outputs    
        res_train = ['TRAIN INFORMATION:',
                    f'Model: {model_name}',
                    f'Number Train Epochs: {epoch}',
                    f'Number Datapoints: {nb_datapoints}',
                    f'Test Size: {test_size}',
                    '\nRESULTS:',
                    f'Runtime: {avg_rt_train/iterations}']
        res_eval = ['EVAL INFORMATION:',
                   f'Model: {model_name}',
                   f'Evaluation Size: {eval_size}', 
                   '\nRESULTS:',
                   f'Pearson-R: {avg_corr/iterations}',
                   f'Standard Deviation: {avg_std/iterations}', 
                   f'MSE: {avg_mse/iterations}',
                   f'Runtime: {avg_rt_eval/iterations}']
        WriteOutput_txt(res_train, path, 'training.txt')
        WriteOutput_txt(res_eval, path, 'evaluation.txt')

        # ADAPTER model
        # measure average runtime training, runtime evaluation, correlation, standard deviation and mse
        avg_rt_train_adapter, avg_rt_eval_adapter, avg_corr_adapter, avg_std_adapter, avg_mse_adapter = 0, 0, 0, 0, 0
        for _ in range(iterations):
            # training
            trainer_adapter, rt_train_adapter = train_metric_adapter(NUM_TRAIN_EPOCHS = epoch,
                                                                     NUMBER_DATAPOINTS = nb_datapoints,
                                                                     TEST_SIZE = test_size,
                                                                     MODEL_NAME = model_name)
            # evaluation
            rt_eval_adapter, corr_adapter, _, std_adapter, mse_adapter = evaluate_metric(trainer_adapter, (epoch, nb_datapoints, test_size), EVAL_SIZE = eval_size, ADAPTER = True, SCATTER_PATH = path)
            # count up measures, return averages later (divided by the number of iterations)
            avg_rt_train_adapter += rt_train_adapter
            avg_rt_eval_adapter += rt_eval_adapter
            avg_corr_adapter += corr_adapter
            avg_std_adapter += std_adapter
            avg_mse_adapter += mse_adapter
        # write outputs
        res_train_adapter = ['TRAIN INFORMATION:',
                            f'Model: {model_name}-adapter',
                            f'Number Train Epochs: {epoch}',
                            f'Number Datapoints: {nb_datapoints}',
                            f'Test Size: {test_size}',
                            '\nRESULTS:',
                            f'Runtime: {avg_rt_train_adapter/iterations}']
        res_eval_adapter = ['EVAL INFORMATION:',
                           f'Model: {model_name}-adapter',
                           f'Evaluation Size: {eval_size}', 
                           '\nRESULTS:',
                           f'Pearson-R: {avg_corr_adapter/iterations}',
                           f'Standard Deviation: {avg_std_adapter/iterations}', 
                           f'MSE: {avg_mse_adapter/iterations}',
                           f'Runtime: {avg_rt_eval_adapter/iterations}']
        WriteOutput_txt(res_train_adapter, path, 'training_adapter.txt')
        WriteOutput_txt(res_eval_adapter, path, 'evaluation_adapter.txt')