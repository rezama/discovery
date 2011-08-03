#! /bin/sh

generations=$1
episodes=$2

gnuplot <<\EOF
set terminal postscript eps color
set xtics $episodes
set grid

set output "graph-champion-trial.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over $generations Generations"
plot "./results-champion-trial.txt" using 1:2 title "Champion Reward"

set output "graph-champion-training.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over $generations Generations"
plot "./results-champion-training.txt" using 1:2 title "Champion Reward"

set output "graph-champion-interval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over $generations Generations"
plot "./results-champion-training-interval.txt" using 1:2 title "Champion Training Reward" with lines, "./results-champion-trial-interval.txt" using 1:2  title "Champion Trial Reward" with lines

set output "graph-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over $generations Generations"
plot "./results-population-interval.txt" using 1:2 title "Population Reward" with lines

EOF 