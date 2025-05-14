# form-data-extraction

To run the scripts run the setup.sh from the main directory that will prepare virtual environment and make all the other scripts executable. The scripts train different models and then run evaluation.

Some of the experiment scripts need .env file with following variables WANDB_API_KEY=wandb_api_key  
HF_TOKEN=hugging_face_token

The repository is structured as follows:

code/ - python scripts to train, inference models and visualize results

data_files/  - data for the new dataset and new dataset in json format

graphs/  - graphs from training included in the thesis

models_tmp/ - default folder for models, they are stored locally as well as pushed on huggingface hub

out/ - outputs of inference files

wandb_csv/ - outputs from wandb for visualization

finetune_lilt_on_forms_dataset.ipynb is from google colab because training LiLT worked there and the same code for some reason did not work on my wsl