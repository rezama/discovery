#! /bin/sh

for w in 0.0 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 0.80 0.85 0.90 0.95 1.00
do
    ~/pypy/bin/pypy $1.py $w
done
