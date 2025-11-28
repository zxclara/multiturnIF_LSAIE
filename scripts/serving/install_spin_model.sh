#!/bin/bash
mkdir ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc

wget https://raw.githubusercontent.com/swiss-ai/model-spinning/refs/heads/main/spin-model.py -O spin-model && chmod 755 spin-model && mv spin-model ~/.local/bin/