set terminal postscript eps color

set xtics 200

set output "compare-champion-training-interval.eps"
set xlabel "Episode"
set ylabel "Champion Training Reward over 15 Generations"
plot "./g15e200t20e50w20/results-champion-training-interval.txt" using 1:2 title "0.50" with lines, \
     "./g15e200t20e75w20/results-champion-training-interval.txt" using 1:2 title "0.75" with lines, \
     "./g15e200t20e90w20/results-champion-training-interval.txt" using 1:2 title "0.90" with lines, \
     "./g15e200t20e95w20/results-champion-training-interval.txt" using 1:2 title "0.95" with lines, \
     "./g15e200t20e97w20/results-champion-training-interval.txt" using 1:2 title "0.97" with lines, \
     "./g15e200t20e99w20/results-champion-training-interval.txt" using 1:2 title "0.99" with lines, \
     "./g15e200t20e100w20/results-champion-training-interval.txt" using 1:2 title "1.00" with lines

set output "compare-champion-trial-interval.eps"
set xlabel "Episode"
set ylabel "Champion Trial Reward over 15 Generations"
plot "./g15e200t20e50w20/results-champion-trial-interval.txt" using 1:2 title "0.50" with lines, \
     "./g15e200t20e75w20/results-champion-trial-interval.txt" using 1:2 title "0.75" with lines, \
     "./g15e200t20e90w20/results-champion-trial-interval.txt" using 1:2 title "0.90" with lines, \
     "./g15e200t20e95w20/results-champion-trial-interval.txt" using 1:2 title "0.95" with lines, \
     "./g15e200t20e97w20/results-champion-trial-interval.txt" using 1:2 title "0.97" with lines, \
     "./g15e200t20e99w20/results-champion-trial-interval.txt" using 1:2 title "0.99" with lines, \
     "./g15e200t20e100w20/results-champion-trial-interval.txt" using 1:2 title "1.00" with lines

set output "compare-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 15 Generations"
plot "./g15e200t20e50w20/results-population-interval.txt" using 1:2 title "0.50" with lines, \
     "./g15e200t20e75w20/results-population-interval.txt" using 1:2 title "0.75" with lines, \
     "./g15e200t20e90w20/results-population-interval.txt" using 1:2 title "0.90" with lines, \
     "./g15e200t20e95w20/results-population-interval.txt" using 1:2 title "0.95" with lines, \
     "./g15e200t20e97w20/results-population-interval.txt" using 1:2 title "0.97" with lines, \
     "./g15e200t20e99w20/results-population-interval.txt" using 1:2 title "0.99" with lines, \
     "./g15e200t20e100w20/results-population-interval.txt" using 1:2 title "1.00" with lines

