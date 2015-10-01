# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact

### BCNN for fine-grained recognition ###

dependencies

The repository contains code using VLFEAT and MATCONVNET to:
+ Extract convolutional feature
+ fine-tuning BCNN models
+ train linear svm classifier on features

please go to MatConvNet and Vlfeat git repositories to download the packages. Our code is built based on MatConvNet version 1.0-beta8. Version 1.0-beta9 also works fine for the speedup by cudnn.
Edit setup.m to link to the packages.