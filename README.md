# PaSCient: Learning multi-cellular representations of single-cell transcriptomics data enables characterization of patient-level disease states



## Installation

To install the related packages for model training, please use:


```
conda env create -f env.yml --name pascient
```

for creating the environment, and then:

```
bash relevant_install.sh
```

for installing the helper packages, and then:

```
pip install -e .
```

for installing the target package.


## Training

To train the model, use the current path, and then run:

```
python cellm/scripts/train_classification_model.py fit --config cellm/configs/disease_classifier.yaml --trainer.logger.save_dir <SAVE_DIRECTORY> --trainer.logger.project <PROJECT_NAME> --trainer.logger.entity <ENTITY_NAME>
```

## Application

Please refer the folder **application** for the experiments we did for disease-state prediction, severity analysis and response prediction.

Please refer the folder **reproduce** for experiments to reproduce the figures we have in this manuscript.

## Contact

If you have any questions, please contact Tianyu Liu (tianyu.liu@yale.edu) or Edward De Brouwer (edward.debrouwer@gmail.com).

## Citation

...
