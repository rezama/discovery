set terminal postscript eps color
set xtics 500
set grid

set output "graph-champions-eval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 15 Generations"
plot "./results-champions-eval.txt" using 1:2 title "Champion Reward"

set output "graph-champions-training.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 15 Generations"
plot "./results-champions-training.txt" using 1:2 title "Champion Reward"

set output "graph-best-champion.eps"
set xlabel "Episode"
set ylabel "Champion's Reward"
plot "./results-best-champion.txt" using 1:2 title "Champion Reward"

set output "graph-champion-interval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 15 Generations"
plot "./results-champions-training-interval.txt" using 1:2 title "Champion Training Reward" with lines, \
     "./results-champions-eval-interval.txt" using 1:2  title "Champion Eval Reward" with lines

set output "graph-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 15 Generations"
plot "./results-population-interval.txt" using 1:2 title "Population Reward" with lines

set output "graph-best-champion-interval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward"
plot "./results-best-champion-interval.txt" using 1:2 title "Best Champion Reward" with lines

