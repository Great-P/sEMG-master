# sEMG-master
simple toolkit of predicting hand motion via sEMG signal

> This toolkit project is currently in its early stages and is a work in progress. It is now a simple representation. 
> 
> Expect periodic updates, but the frequency and timing of these updates are subject to change
## Usage

### Dataset
> Note: This work is built upon the Ninapro DB2 dataset; for compatibility with other datasets, additional code adaptations may be required.
1. Put the mat data to ./Data/rawdata
2. Extract feature with ./utils/DataProcess/FeatureExtract.py

### Direct Training
./Excutors/Trainer.py 

### Transfer/Lifelong(Unpublished Now)
> Support LCSN only. 
> 
> This part will be unveiled shortly...
 

All under Excutors
* EWC:
  EWC_LCSN.py
* LwF:
  LearningWithoutForgettingLCSN.py
* DomainAdaption:
  DomainAdaption.py
* FineTuning:
  FineTuningLSCN.py

* LifelongMethod:LLTransferLCSN.py