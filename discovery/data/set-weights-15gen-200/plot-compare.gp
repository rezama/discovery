set terminal postscript eps color

set yrange  [-0.1:0.50]
set key left top

set output "graph-champion-interval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 15 Generations"
plot "./run2/15gen-200ep-optweights1.0/results-champion-interval.txt" using 1:2 title "Optimistic 1.0 Run 1" with lines, "./run2/15gen-200ep-optweights0.5/results-champion-interval.txt" using 1:2 title "Optimistic 0.5 Run 1" with lines, "./run2/15gen-200ep-zeroweights/results-champion-interval.txt" using 1:2 title "Zero Run 1" with lines, "./run3/15gen-200ep-optweights1.0/results-champion-interval.txt" using 1:2 title "Optimistic 1.0 Run 2" with lines, "./run3/15gen-200ep-optweights0.5/results-champion-interval.txt" using 1:2 title "Optimistic 0.5 Run 2" with lines, "./run3/15gen-200ep-zeroweights/results-champion-interval.txt" using 1:2 title "Zero Run 2" with lines

set output "graph-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 15 Generations"
plot "./run2/15gen-200ep-optweights1.0/results-population-interval.txt" using 1:2 title "Optimistic 1.0 Run 1" with lines, "./run2/15gen-200ep-optweights0.5/results-population-interval.txt" using 1:2 title "Optimistic 0.5 Run 1" with lines, "./run2/15gen-200ep-zeroweights/results-population-interval.txt" using 1:2 title "Zero Run 1" with lines, "./run3/15gen-200ep-optweights1.0/results-population-interval.txt" using 1:2 title "Optimistic 1.0 Run 2" with lines, "./run3/15gen-200ep-optweights0.5/results-population-interval.txt" using 1:2 title "Optimistic 0.5 Run 2" with lines, "./run3/15gen-200ep-zeroweights/results-population-interval.txt" using 1:2 title "Zero Run 2" with lines
