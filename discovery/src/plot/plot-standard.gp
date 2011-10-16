set terminal postscript eps color

set output "graph-standard-training.eps"
set xlabel "Episode"
set ylabel "Average Reward"
plot "./results-standard-training.txt" using 1:2 title "Average Training Reward over 20 trials"

set output "graph-standard-training-interval.eps"
set xlabel "Episode"
set ylabel "Average Reward"
plot "./results-standard-training-interval.txt" using 1:2 title "Average Training Reward over 20 trials" with lines

set output "graph-standard-eval.eps"
set xlabel "Episode"
set ylabel "Average Reward"
plot "./results-standard-eval.txt" using 1:2 title "Average Eval Reward over 20 trials"

set output "graph-standard-eval-interval.eps"
set xlabel "Episode"
set ylabel "Average Reward"
plot "./results-standard-eval-interval.txt" using 1:2 title "Average Eval Reward over 20 trials" with lines
