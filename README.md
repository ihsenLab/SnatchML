## Overview
This repository contains the official implementation for the paper "Model for Peanuts: Hijacking ML models without Training Access is Possible." In this paper, we explore the threat of model hijacking in machine learning (ML) models, where an adversary aims to repurpose a victim model to perform a different task than its original one, without access to the model's training phase.

## Features
- Implementation of SnatchML algorithm
- Experimental setups and results for various hijacking scenarios
- Datasets used for evaluation

## Datasets:
- **CK+ Dataset**: Extended Cohn-Kanade dataset for emotion recognition.
- **Olivetti Faces**: Dataset for facial recognition.
- **Celebrity Dataset**: Facial images for various celebrities, augmented with emotion-specific images.
- **Synthetic Dataset**: Synthetic dataset generated using MakeHuman for emotion and identity recognition.
- **UTKFace Dataset**: Collection of facial images labeled with age, gender, and ethnicity.
- **Chest X-Ray Images (Pneumonia)**: Dataset containing X-ray images categorized into 'Pneumonia' and 'Normal' classes, further labeled into 'Viral' and 'Bacterial' subcategories.

## Datasets Preparation:
Download the datasets provided in the following Google Drive link and upload them to the './datasets' folder before running tany script.
Link to the datasets: https://drive.google.com/drive/folders/1igivoksoUquXVbbV7W_B3Yjj3o4qpqzc?usp=sharing

## Hijacking ER Models:
To run the hijacking attack on ER models for re-identification on CK+ or biometric identification on Olivetti, Celebrity, and Synthetic datasets, run the following script by specifying: ER model architecture, hijacking dataset, and attack setting (white-box|black-box):
```shell
$ python hijack_er.py --setting [black|white] --model [architecture] --hijack-dataset [target hijacking dataset]
```

Example: For hijacking an ER model with MobileNet architecture for user identification on the Olivetti dataset under white-box setting:
```shell
$ python hijack_er.py --setting white --model mobilenet --hijack-dataset olivetti
```

## Hijacking PDD Models:
To run the hijacking attack on PDD models for recongnizing the type of the pneumonia infection (bacterial or viral) on the Chest X-ray dataset, run the following script by specifying: ER model architecture and attack setting (white-box|black-box):
```shell
$ python hijack_pneu.py --setting [black|white] --model [architecture]
```

Example: For hijacking a PDD model with ResNet-9 architecture for recognizing the pneumonia infection on the Chest X-ray dataset under black-box setting:
```shell
$ python hijack_penu.py --setting black --model resnet
```

## Cross-attribute Hijacking:
To run the hijacking attack on human attribute prediction models (e.g., for age, gender, and ethnicity) on the UTK dataset, run the following script by specifying: ER model architecture, original task, hijacking attack, and attack setting (white-box|black-box):
```shell
$ python hijack_utk.py --setting [black|white] --model [architecture] --original-task [original task] --hijack-task [Hijacking task]
```

Example: For hijacking an age estimation model with 2D-CNN architecture for ethnicity/race recognition on the UTK dataset under white-box setting:
```shell
$ python hijack_utk.py --setting white --model simple --original-task age --hijack-task race
```

## Over-parametrized Models Study:
To reproduce our results on overparametrized models (Sections 9.1 and 10.2) of the paper, we provide automated scripts for each case as following (results will be auto-saved in ./results folder):
```shell
$ bash auto_test_er.sh $[hijacking dataset: ck, olivetti, celebrity, synthetic]
```

```shell
$ bash auto_test_pneu.sh
```

```shell
$ bash auto_test_utk.sh $[original task: age, gender, race] $[hijacking task: age, gender, race]
```

## Meta-unlearning experiments:
To reproduce our results on meta-unlearning models (Section 10.1) of the paper, run the following scripts for PDD and ER cases:
```shell
$ python unlearn_pneu.py --setting [black|white] --model [architecture] --alpha [0, 1] --beta [0, 1]
```

```shell
$ python unlearn_er.py --setting white --model mobilenet --hijack-dataset olivetti --alpha [0, 1] --beta [0, 1]
```
