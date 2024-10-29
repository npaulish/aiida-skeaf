#!/bin/bash
#
# screen -list
# screen -S npaulish
# detach: Ctrl+a d
# screen -r npaulish

while true; do

./submit_aiida_shell.py \
        -r \
        --max_concurrent 48 \

sleep 30s

done
