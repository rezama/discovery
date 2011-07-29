set terminal postscript eps color

set output "graph-champion-interval.eps"
set xlabel "Episode"
set ylabel "Champion's Reward over 15 Generations"
plot "./eta1.0/results-champion-interval.txt" using 1:2 title "Eta 1.0" with lines, "./eta0.97/results-champion-interval.txt" using 1:2 title "Eta 0.97" with lines, "./eta0.95/results-champion-interval.txt" using 1:2 title "Eta 0.95" with lines, "./eta0.90/results-champion-interval.txt" using 1:2 title "Eta 0.90" with lines

set output "graph-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 15 Generations"
plot "./eta1.0/results-population-interval.txt" using 1:2 title "Eta 1.0" with lines, "./eta0.97/results-population-interval.txt" using 1:2 title "Eta 0.97" with lines, "./eta0.95/results-population-interval.txt" using 1:2 title "Eta 0.95" with lines, "./eta0.90/results-population-interval.txt" using 1:2 title "Eta 0.90" with lines

