#! /bin/sh

trials=$1
episodes=$2

gnuplot <<\EOF
set terminal postscript eps color

set output "graph-standard.eps"
set xlabel "Episode"
set ylabel "Average Reward"
plot "./results-standard.txt" using 1:2 title "Average Reward over $trials trials" with lines

set output "graph-standard-interval.eps"
set xlabel "Episode"
set ylabel "Average Reward"
plot "./results-standard-interval.txt" using 1:2 title "Average Reward over $trials trials" with lines

EOF
 