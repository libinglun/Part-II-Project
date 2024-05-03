# Part-II-Project
Infinited HMM (HDP-HMM) model for PoS tagging induction

## How to run
### Cloning the Repository
Cloning the repository from GitHub:
```angular2html
git clone https://github.com/libinglun/Part-II-Project.git
```
### Setting up a virtual environment
```angular2html
python -m venv venv
source venv/bin/activate
```
### Installing the Project
```angular2html
pip install -r requirements.txt
pip install .
```
For an editable install, which allows you to modify the code and see changes without reinstalling:
```
pip install -e .
```
### Running the project 

#### HMM Synthetic Test 
To train the model on an HMM synthetic test, use the command (example):
```angular2html
hmm-test -mode train -iter 10 -noise 0.5 -states 10 -obs 500 -size 5000
```
Where `-mode` chooses between `train` mode and `resume` mode; `-iter` sets the iterations. The trailing four 
arguments specify the `noise-level`, `num of states`, `num of observations`, `dataset size`, respectively. 

To resume from a trained state of model, use the command (example):
```angular2html
hmm-test -mode resume -state "path to the state saved in the result folder" -iter 10 -noise 0.5 -states 10 -obs 500 -size 5000
```
Where in the `resume` mode, the model will load the saved state of previous training and resume training on the dataset. 
> Note that the state would be stored in an `npz` file in the result folder. DON'T include the .npz suffix in the path.

#### Natural Language Test
To train the model on a natural language dataset, use the command:
```angular2html
lang-test -mode train -iter 10 -noise 0.5 -name PTB  
```
Where `-mode` and `-iter` are similar arguments as in the HMM test. `-noise` (optional) specifies the noise-level of the dataset, 
and `-name` specifies the name of the dataset to be trained on. 

> Currently only supports Penn-treebank as `PTB` and Childes as `Childes`

To resume from the a trained state of model, use the command:
```angular2html
lang-test -mode resume -state "path to the state saved in the result folder" -iter 10 -noise 0.5 -name PTB  
```