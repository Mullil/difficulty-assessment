# difficulty-assessment

<<<<<<< HEAD
=======
Datasetti
6220 instanssia
splitataan 80/20:
train 4976 + augmented n. 5000 (2500 konekääntäminen + 2500 kielimalli)
validation 1244

Data distribution:
1:      181 = 2,9%
1.5:    256 = 4,1%
2:      446 = 7,2%
2.5:    56 = 0,9%
3:      2115 = 34%
3.5:    1377 = 22,1%
4:      579 = 9,3%
5:      370 = 5,9%
5.5:    732 = 11,8%
6:      107 = 1,7% 


Suunnitelma:

Data-augmentation
- Konekääntäminen HuggingFacen CEFR datasetin englannin ja viron instansseista.
- Kiertoilmaukset kielimallin avulla

Malli:
TurkuNLP/bert-base-finnish-cased-v1

Luokittelu:
Regressio

Hyperparametrit optunalle:
weight-decay
warmup ratio
classification lr
encoder lr
attention dropout
epochs
datasetit


1. datasetin koostaminen
- train/validation stratified split
- konekääntäminen CEFR datasetistä
- kiertoilmaukset kielimallin avulla

2. treenausskripti
3. evaluaatioskripti
4. raportti
>>>>>>> 1e14d8daf6adce6e574ee9bf78ec097a96582d63
