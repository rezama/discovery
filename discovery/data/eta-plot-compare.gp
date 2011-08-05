set terminal postscript eps color

set xtics 200

set output "eta-compare-champion-training-interval.eps"
set xlabel "Episode"
set ylabel "Champion Training Reward over 15 Generations"
plot "./g15e200t200e25w0/results-champion-training-interval.txt" using 1:2 title "eta 0.25" with lines, \
     "./g15e200t200e50w0/results-champion-training-interval.txt" using 1:2 title "eta 0.50" with lines, \
     "./g15e200t200e75w0/results-champion-training-interval.txt" using 1:2 title "eta 0.75" with lines, \
     "./g15e200t200e90w0/results-champion-training-interval.txt" using 1:2 title "eta 0.90" with lines, \
     "./g15e200t200e95w0/results-champion-training-interval.txt" using 1:2 title "eta 0.95" with lines, \
     "./g15e200t200e97w0/results-champion-training-interval.txt" using 1:2 title "eta 0.97" with lines, \
     "./g15e200t200e99w0/results-champion-training-interval.txt" using 1:2 title "eta 0.99" with lines, \
     "./g15e200t200e100w0/results-champion-training-interval.txt" using 1:2 title "eta 1.00" with lines

set output "eta-compare-champion-trial-interval.eps"
set xlabel "Episode"
set ylabel "Champion Trial Reward over 15 Generations"
plot "./g15e200t200e25w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.25" with lines, \
     "./g15e200t200e50w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.50" with lines, \
     "./g15e200t200e75w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.75" with lines, \
     "./g15e200t200e90w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.90" with lines, \
     "./g15e200t200e95w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.95" with lines, \
     "./g15e200t200e97w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.97" with lines, \
     "./g15e200t200e99w0/results-champion-trial-interval.txt" using 1:2 title "eta 0.99" with lines, \
     "./g15e200t200e100w0/results-champion-trial-interval.txt" using 1:2 title "eta 1.00" with lines

set output "eta-compare-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 15 Generations"
plot "./g15e200t200e25w0/results-population-interval.txt" using 1:2 title "eta 0.25" with lines, \
     "./g15e200t200e50w0/results-population-interval.txt" using 1:2 title "eta 0.50" with lines, \
     "./g15e200t200e75w0/results-population-interval.txt" using 1:2 title "eta 0.75" with lines, \
     "./g15e200t200e90w0/results-population-interval.txt" using 1:2 title "eta 0.90" with lines, \
     "./g15e200t200e95w0/results-population-interval.txt" using 1:2 title "eta 0.95" with lines, \
     "./g15e200t200e97w0/results-population-interval.txt" using 1:2 title "eta 0.97" with lines, \
     "./g15e200t200e99w0/results-population-interval.txt" using 1:2 title "eta 0.99" with lines, \
     "./g15e200t200e100w0/results-population-interval.txt" using 1:2 title "eta 1.00" with lines

