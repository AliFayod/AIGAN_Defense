# AIGan PyTorch

Unofficial implementation of AIGan from paper "AI-GAN: Attack-Inspired Generation of Adversarial Examples".

Code is based on mathcbc/advGAN_pytorch and ctargon/AdvGAN-tf with my modifications.

## training the target model

`python3 train_target_model.py`  


## training the AIGAN

`python3 main.py`

## testing adversarial examples

`python3 ad_test.py`

# AIGAN_Defense

This project is a defense against an attack (add nois to the data before came through input) on a neural network trained to classify mnist data In addition to using the other defenses prepackaged in the various defense libraries that I used in comparing the defense is created in this project with the other defenses, The attack used is AIGAN that was introduced in the early months of 2021

## Build our Defense 

`python3 defenseAIGAN.py`

## Different Defense 

`python3 L1PGDAttack_Defence`
`python3 L2PGDAttack_Defence`
`python3 LinfPGDAttack_Defence`

> etc
