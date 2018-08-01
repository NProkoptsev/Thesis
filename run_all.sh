#!/bin/bash

set -euo pipefail

cecho() {
    echo "\033[0;31m$1\033[0m"
}

source venv/bin/activate

cecho "Training VAR... It's pretty fast"
python run.py train var var

cecho "Training LSTM... It takes longer"
python run.py train lstm lstm

cecho "Training IndRNN... Please, be patient"
python run.py train indrnn indrnn

cecho "Training GCGRU... Please, be patient"
python run.py train gcgru gcgru

cecho "Evaluating accuracy: VAR"
python run.py infer var var

cecho "Evaluating accuracy: LSTM"
python run.py infer lstm lstm

cecho "Evaluating accuracy: IndRNN"
python run.py infer indrnn indrnn

cecho "Evaluating accuracy: GCGRU"
python run.py infer gcgru gcgru
