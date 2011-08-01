set terminal postscript eps color

set key left

set output "graph-champion-interval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 15 Generations"
plot "./optweights1.0/results-champion-interval.txt" using 1:2 title "Optimistic 1.0" with lines, "./optweights0.75/results-champion-interval.txt" using 1:2 title "Optimistic 0.75" with lines, "./optweights0.5/results-champion-interval.txt" using 1:2 title "Optimistic 0.5" with lines, "./optweights0.25/results-champion-interval.txt" using 1:2 title "Optimistic 0.25" with lines, "./zeroweights/results-champion-interval.txt" using 1:2 title "Zero" with lines

set output "graph-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 15 Generations"
plot "./optweights1.0/results-population-interval.txt" using 1:2 title "Optimistic 1.0" with lines, "./optweights0.75/results-population-interval.txt" using 1:2 title "Optimistic 0.75" with lines, "./optweights0.5/results-population-interval.txt" using 1:2 title "Optimistic 0.5" with lines, "./optweights0.25/results-population-interval.txt" using 1:2 title "Optimistic 0.25" with lines, "./zeroweights/results-population-interval.txt" using 1:2 title "Zero" with lines
