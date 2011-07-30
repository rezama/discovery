set terminal postscript eps color
set xtics 500
set grid

set output "graph-champion.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 10 Generations"
plot "./results-champion.txt" using 1:2  title "Champion Reward"

set output "graph-champion-interval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 10 Generations"
plot "./results-champion-interval.txt" using 1:2  title "Champion Reward" with lines

set output "graph-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 10 Generations"
plot "./results-population-interval.txt" using 1:2  title "Population Reward" with lines
