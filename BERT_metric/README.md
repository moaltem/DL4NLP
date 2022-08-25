## Analyzing the Effect of Adapters on the Efficiency of Natural Language Generation Evaluation Metrics

### Research Project in the scope of ‘Deep Learning for Natural Language Processing (392197)’ at University of Bielefeld
#### by Moritz Altemeyer and Philipp Hege
------------------------------

This repository contains the code of the BERT experiment. In detail, it consists of the following:

- Folder **__pycache__**: Usual Python caches created automatically during execution.
- Folder **data**: Contains the data used for the project: WMT17. One train data set used for training and testing within training, one test data set used for evaluation.
- Folder **model_checkpoints**: Contains the model checkpoints of different model configurations generated while execution; Contains config json files for model and tokenizer, bins and so on. Consists of several folders (in the style of: \<model name\>\_\_\<number epochs\>\_\_\<number datapoints\>_\_\<test size\>) for the corresponding model configurations. 
- Folder **results**: Contains the results of different model configurations generated while execution. Consists of several folders (in the style of: \<model name\>\_\_\<number epochs\>\_\_\<number datapoints\>\_\_\<test size\>), each containing for both, the standard as well as the adapter model a .pdf file with the predictions scatter plot, a evaluation.txt containing the results of the evaluation and a training.txt containing the results of the training process. In addition, **results** contains some checkpoint and run folders automatically generated during execution.
- Document **evaluate.py**: Evaluation of the BERT metric.
- Document **main.py**: Main .py file, starts the whole training and evaluation process. The important settings are defined by global variables in the beginning of the file. Execute the file via python interpreter directly or use bash. Regarding latter, navigate in the folder where the main.py file houses and use ‘python main.py’ (alternatively: ‘python3 main.py’)  bash command to start.
- Document **tools.py**: Contains necessary assisting functions for basic processes used in BERT metric.
- Document **train.py**: Training of the BERT metric.
--------------------------------

**NOTE**: Start training the metric using main.py (see above).
