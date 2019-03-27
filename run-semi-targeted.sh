#!/bin/bash

# blackbox, attack, substitute
python -m blackbox.semi_targeted 1 fgsm 1 > semitargeted-1-fgsm-1.txt
python -m blackbox.semi_targeted 2 fgsm 1 > semitargeted-2-fgsm-1.txt
python -m blackbox.semi_targeted 3 fgsm 1 > semitargeted-3-fgsm-1.txt
python -m blackbox.semi_targeted 4 fgsm 1 > semitargeted-4-fgsm-1.txt

#python -m blackbox.semi_targeted 1 fgsm 2 > semitargeted-1-fgsm-2.txt
#python -m blackbox.semi_targeted 2 fgsm 2 > semitargeted-2-fgsm-2.txt
#python -m blackbox.semi_targeted 3 fgsm 2 > semitargeted-3-fgsm-2.txt
#python -m blackbox.semi_targeted 4 fgsm 2 > semitargeted-4-fgsm-2.txt

#python -m blackbox.semi_targeted 1 fgsm 3 > semitargeted-1-fgsm-3.txt
#python -m blackbox.semi_targeted 2 fgsm 3 > semitargeted-2-fgsm-3.txt
#python -m blackbox.semi_targeted 3 fgsm 3 > semitargeted-3-fgsm-3.txt
#python -m blackbox.semi_targeted 4 fgsm 3 > semitargeted-4-fgsm-3.txt

#python -m blackbox.semi_targeted 1 fgsm 4 > semitargeted-1-fgsm-4.txt
#python -m blackbox.semi_targeted 2 fgsm 4 > semitargeted-2-fgsm-4.txt
#python -m blackbox.semi_targeted 3 fgsm 4 > semitargeted-3-fgsm-4.txt
#python -m blackbox.semi_targeted 4 fgsm 4 > semitargeted-4-fgsm-4.txt

#python -m blackbox.semi_targeted 1 cw 1 > semitargeted-1-cw-1.txt
#python -m blackbox.semi_targeted 2 cw 1 > semitargeted-2-cw-1.txt
#python -m blackbox.semi_targeted 3 cw 1 > semitargeted-3-cw-1.txt
#python -m blackbox.semi_targeted 4 cw 1 > semitargeted-4-cw-1.txt

#python -m blackbox.semi_targeted 1 cw 2 > semitargeted-1-cw-2.txt
#python -m blackbox.semi_targeted 2 cw 2 > semitargeted-2-cw-2.txt
#python -m blackbox.semi_targeted 3 cw 2 > semitargeted-3-cw-2.txt
#python -m blackbox.semi_targeted 4 cw 2 > semitargeted-4-cw-2.txt

#python -m blackbox.semi_targeted 1 cw 3 > semitargeted-1-cw-3.txt
#python -m blackbox.semi_targeted 2 cw 3 > semitargeted-2-cw-3.txt
#python -m blackbox.semi_targeted 3 cw 3 > semitargeted-3-cw-3.txt
#python -m blackbox.semi_targeted 4 cw 3 > semitargeted-4-cw-3.txt

python -m blackbox.semi_targeted 1 cw 4 > semitargeted-1-cw-4.txt
python -m blackbox.semi_targeted 2 cw 4 > semitargeted-2-cw-4.txt
python -m blackbox.semi_targeted 3 cw 4 > semitargeted-3-cw-4.txt
python -m blackbox.semi_targeted 4 cw 4 > semitargeted-4-cw-4.txt


