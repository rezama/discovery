#! /bin/sh

#for w in 0.0 0.05 0.10 0.20 0.25 0.50 1.00
#do
#    ~/pypy/bin/pypy $1.py $w
#done

for eta in 0.25 0.50 0.75 0.90 0.95 0.97 0.99 1.00
do
    ~/pypy/bin/pypy $1.py $eta
done
