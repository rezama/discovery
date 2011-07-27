set terminal postscript eps color

set output "graph-champion-interval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 10 Generations"
plot "./15gen-200ep-optweights1.0/results-champion-interval.txt" using 1:2 title "Optimistic 1.0" with lines, "./15gen-200ep-optweights0.5/results-champion-interval.txt" using 1:2 title "Optimistic 0.5" with lines, "./15gen-200ep-zeroweights/results-champion-interval.txt"   using 1:2 title "Zero" with lines

set output "graph-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 10 Generations"
plot "./15gen-200ep-optweights1.0/results-population-interval.txt" using 1:2 title "Optimistic 1.0" with lines, "./15gen-200ep-optweights0.5/results-population-interval.txt" using 1:2 title "Optimistic 0.5" with lines, "./15gen-200ep-zeroweights/results-population-interval.txt"   using 1:2 title "Zero" with lines
