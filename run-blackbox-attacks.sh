#!/bin/bash

# black box, attack, substitute
python -m blackbox.transfer_age 1 fgsm 1 > bbox-1-fgsm-1.txt
python -m blackbox.transfer_age 1 fgsm 2 > bbox-1-fgsm-2.txt
python -m blackbox.transfer_age 1 fgsm 3 > bbox-1-fgsm-3.txt
python -m blackbox.transfer_age 1 fgsm 4 > bbox-1-fgsm-4.txt

python -m blackbox.transfer_age 2 fgsm 1 > bbox-2-fgsm-1.txt
python -m blackbox.transfer_age 2 fgsm 2 > bbox-2-fgsm-2.txt
python -m blackbox.transfer_age 2 fgsm 3 > bbox-2-fgsm-3.txt
python -m blackbox.transfer_age 2 fgsm 4 > bbox-2-fgsm-4.txt

python -m blackbox.transfer_age 3 fgsm 1 > bbox-3-fgsm-1.txt
python -m blackbox.transfer_age 3 fgsm 2 > bbox-3-fgsm-2.txt
python -m blackbox.transfer_age 3 fgsm 3 > bbox-3-fgsm-3.txt
python -m blackbox.transfer_age 3 fgsm 4 > bbox-3-fgsm-4.txt

python -m blackbox.transfer_age 4 fgsm 1 > bbox-4-fgsm-1.txt
python -m blackbox.transfer_age 4 fgsm 2 > bbox-4-fgsm-2.txt
python -m blackbox.transfer_age 4 fgsm 3 > bbox-4-fgsm-3.txt
python -m blackbox.transfer_age 4 fgsm 4 > bbox-4-fgsm-4.txt


python -m blackbox.transfer_age 1 cw 1 > bbox-1-cw-1.txt
python -m blackbox.transfer_age 1 cw 2 > bbox-1-cw-2.txt
python -m blackbox.transfer_age 1 cw 3 > bbox-1-cw-3.txt
python -m blackbox.transfer_age 1 cw 4 > bbox-1-cw-4.txt

python -m blackbox.transfer_age 2 cw 1 > bbox-2-cw-1.txt
python -m blackbox.transfer_age 2 cw 2 > bbox-2-cw-2.txt
python -m blackbox.transfer_age 2 cw 3 > bbox-2-cw-3.txt
python -m blackbox.transfer_age 2 cw 4 > bbox-2-cw-4.txt

python -m blackbox.transfer_age 3 cw 1 > bbox-3-cw-1.txt
python -m blackbox.transfer_age 3 cw 2 > bbox-3-cw-2.txt
python -m blackbox.transfer_age 3 cw 3 > bbox-3-cw-3.txt
python -m blackbox.transfer_age 3 cw 4 > bbox-3-cw-4.txt

python -m blackbox.transfer_age 4 cw 1 > bbox-4-cw-1.txt
python -m blackbox.transfer_age 4 cw 2 > bbox-4-cw-2.txt
python -m blackbox.transfer_age 4 cw 3 > bbox-4-cw-3.txt
python -m blackbox.transfer_age 4 cw 4 > bbox-4-cw-4.txt
