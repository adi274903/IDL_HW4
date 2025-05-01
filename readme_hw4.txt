## HW3P2 IDL
- Author: Aditya Sannabhadti
- Andrew: asannabh
- Kaggle: adisann


## Description:
The given HW made us make a Transformer model for ASR task where lowering CER was the objective. 
Using the professor given dataset a classification needed to be made for the multiple faces using the trained model.

## Requirements:
Given in the requirements.txt file

## Dataset:
Dataset used was the  which the professor provided and can be found in given Kaggle competition:
https://www.kaggle.com/competitions/11785-s25-hw4p2-asr


## Model
Describe the model architecture.
- Type of model was the first ever transformer model utilizing a pre-layer norm for encoder and post layer norm for decoder, contained multihead attention and cross attention with GELU function, approximately 29.9M parameters
- Number of layers was approximately 33 layers 
- Activation functions used here was GELU, which was extremely effective.
- Loss function and optimizer:
	CTC Loss, Adam W and Cosine Annealing was used for given task.
- Model Train time: 5-6 hours, due to attention and use of both reduction methods (conv,lstm)

## Experimentation:
For given Hw4P2 approximately 51 ablations were run until CER of ~10.66 was achieved. To summarize the given studies:

- (1-22 Ablations):
	. Ablation studies were conducted using the pre given model with slightly changing the number of layers, heads and changing the type of reduction used.
	. Approximately the networks between 6.6M and 26.5M parameters.
	. Given reached approximately 22 CER.

- (22-40 Ablations): 
	. These models tried to incorporate more number of encoder layers and less decoder layers, which led to better performance and also more reduction was used from 2 to 4.
	. The accuracy increased to approximately 15 CER. 
	. The models usually spanned between 15.2M to 29.6M parameters.
	. SGU with gMLP was also used to test its effectiveness, which did not prove to be helpful.
	. Few runs also utilized the pu feature, which was not successful as well.

- (40-51 Ablations):
	. These models tried to incorporate thehigher encoder layers, and a baseline was detected to be 12 with decoder layers being 6. Attention heads were 4 and d_model was 256. 
	. This proved to be the most optimal setup with tie_reduction being 4, which sped up the training as well with approximately 10min per epoch.
	. Models spanned approximately 19.8M - 29.9M parameters
	


Tips after the studies for given dataset:
- Use both as time reduction strategy and make the reduction to 4 as it proved to work effectively.
- Initially running it for 20 epochs lead to diminishing returns and after increasing the epochs to more than 30, better results were achieved in terms of accuracy.
- Pre-Layer Norm should not be changed at all.
- A100 GPU is the only way to train these models because of the computations carried out by the model.
- A smaller batch size of 8 and 16 would be recommended initially, but after 30 epochs please change it to higher sizes of 32 or 64. Helps the model fine tune more to the dataset.
- Dataset should have multiple different types of mask every epoch to help model generalize better.


## Training
Instructions on how to train the model.
- For training, just run the attached .ipynb file in google colab and ensure a "chekpoints" folder is made for given runtime.
- Also ensure that the given Wandb part of ipynb is renamed for the new training model for checkpoints.  

## Evaluation
How to test the trained model.
- Directly run the code after the "Iterate over the number of epochs to train and evaluate your model" section of the ipynb file, ensuring that given "checkpoints" has the wandb pth file of the model for testing or by loading the given run id:
 . model_basic_1.32 :  "nwnxjtt5"


## Inference
Steps to use the trained model for predictions.
- Predictions can be checked directly through the generated "submission.csv" file after running the .ipynb file.

## Notes
- The given ipynb notebook will work but for testing the model I would recommend to load the architecture and weights from the given "model_basic_1.32" runs .
- The wandb link for given ablation studies is as follows:
	https://wandb.ai/adisann20025-carnegie-mellon-university/hw4p2?


## Acknowledgment
- I would like to acknowledge my mentor Purusottam Samal who suggested us the idea to check out multiple research papers before starting the assignment, implementing multiple permutations per epoch, utilizing multiple masking per data and for implementing them. Without his help, getting a CER less than 10.66 would not have been possible for the whole study group.

