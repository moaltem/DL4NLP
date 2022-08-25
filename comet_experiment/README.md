## Analyzing the Effect of Adapters on the Efficiency of Natural Language Generation Evaluation Metrics

### Research Project in the scope of ‘Deep Learning for Natural Language Processing (392197)’ at University of Bielefeld
#### by Moritz Altemeyer and Philipp Hege
---------------------------

This repository contains the code of the COMET experiment. In detail, it consists of the following:

— Folder **comet_train**: Files and data used for training.

— Folder **comet_test**: Files and data used for testing.

— Folder **encoders**: Modified encoder folder for the unbabel-comet library.

— Document **comet_environment.yml**: Conda environment file containing all necessary libraries.

---------------------------

1. For running our experiments set a conda environment according to the comet_environment.yml.

2. Replace the **encoder** folder in the unbabel-comet library of the created environment with the modified one located in this repository.

3. Train COMET framework:
	- Set the **comet_test** folder as the working directory in bash using the cd command.
	- Run the command 'comet-train --cfg configs/models/<model_name>.yaml' and replace <model_name> by the corresponding model to train.
		- Available models: minilm, minilm_adapters, xlmr and xlmr_adapters.
	- The folder **lighning_logs** in the **comet_test** folder includes the checkpoints of trained models and corresponding TFEvent files. 

4. Test COMET framework:
	- The file **test_script.ipynb** in the folder **comet_test** provides a detailed description for the evaluation of the trained metrics.
