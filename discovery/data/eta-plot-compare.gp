set terminal postscript eps color

set xtics 500
set grid
set yrange [-.3:0.75]

set output "eta-compare-champion-training-interval.eps"
set xlabel "Episode"
set ylabel "Champion Training Reward over 10 Generations"
plot "./g10e500t200e25w0/results-champion-training-interval.txt" using 1:2 title "eta 0.25" with lines, \
     "./g10e500t200e95w0/results-champion-training-interval.txt" using 1:2 title "eta 0.95" with lines, \
     "./g10e500t200e97w0/results-champion-training-interval.txt" using 1:2 title "eta 0.97" with lines, \
     "./g10e500t200e99w0/results-champion-training-interval.txt" using 1:2 title "eta 0.99" with lines, \
     "./g10e500t200e100w0/results-champion-training-interval.txt" using 1:2 title "eta 1.00" with lines

set output "eta-compare-champion-trial-interval.eps"
set xlabel "Episode"
set ylabel "Champion Trial Reward over 10 Generations"
plot "./g10e500t200e25w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.25" with lines, \
     "./g10e500t200e95w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.95" with lines, \
     "./g10e500t200e97w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.97" with lines, \
     "./g10e500t200e99w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.99" with lines, \
     "./g10e500t200e100w0/results-champion-trial-interval.txt" using 1:2 title "eta 1.00" with lines

set output "eta-compare-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 10 Generations"
plot "./g10e500t200e25w0/results-population-interval.txt" using 1:2 title "eta 0.25" with lines, \
     "./g10e500t200e95w0/results-population-interval.txt" using 1:2 title "eta 0.95" with lines, \
     "./g10e500t200e97w0/results-population-interval.txt" using 1:2 title "eta 0.97" with lines, \
     "./g10e500t200e99w0/results-population-interval.txt" using 1:2 title "eta 0.99" with lines, \
     "./g10e500t200e100w0/results-population-interval.txt" using 1:2 title "eta 1.00" with lines

