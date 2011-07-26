set terminal postscript eps color

set output "graph-champion-interval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 10 Generations"
plot "./10gen-200ep-multi-ballfixed-optweights/results-champion-interval.txt" using 1:2  title "Optimistic Initialization" with lines, "./10gen-200ep-multi-ballfixed-zeroweights/results-champion-interval.txt"   using 1:2  title "Pessimistic (Zero) Initialization" with lines

set output "graph-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 10 Generations"
plot "./10gen-200ep-multi-ballfixed-optweights/results-population-interval.txt" using 1:2  title "Optimistic Initialization" with lines, "./10gen-200ep-multi-ballfixed-zeroweights/results-population-interval.txt"   using 1:2  title "Pessimistic (Zero) Initialization" with lines
