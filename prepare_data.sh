#!/bin/bash

cd data
mkdir -p Adult
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -o Adult/adult.data
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -o Adult/adult.test

mkdir -p COMPAS
curl https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv -o COMPAS/compas-scores-two-years.csv

mkdir -p LSAC
curl http://www.seaphe.org/databases/LSAC/LSAC_SAS.zip -o LSAC/LSAC_SAS.zip
unzip -o LSAC/LSAC_SAS.zip -d LSAC

cd -

python prepare_data.py

