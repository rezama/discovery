set terminal postscript eps color

set xtics 200

set output "w-compare-champion-training-interval.eps"
set xlabel "Episode"
set ylabel "Champion Training Reward over 15 Generations"
plot "./g15e200t100e99w0/results-champion-training-interval.txt" using 1:2 title "w 0.00" with lines, \
     "./g15e200t100e99w5/results-champion-training-interval.txt" using 1:2 title "w 0.05" with lines, \
     "./g15e200t100e99w10/results-champion-training-interval.txt" using 1:2 title "w 0.10" with lines, \
     "./g15e200t100e99w20/results-champion-training-interval.txt" using 1:2 title "w 0.20" with lines, \
     "./g15e200t100e99w25/results-champion-training-interval.txt" using 1:2 title "w 0.25" with lines, \
     "./g15e200t100e99w50/results-champion-training-interval.txt" using 1:2 title "w 0.50" with lines, \
     "./g15e200t100e99w100/results-champion-training-interval.txt" using 1:2 title "w 1.00" with lines

set output "w-compare-champion-trial-interval.eps"
set xlabel "Episode"
set ylabel "Champion Trial Reward over 15 Generations"
plot "./g15e200t100e99w0/results-champion-trial-interval.txt" using 1:2 title "w 0.00" with lines, \
     "./g15e200t100e99w5/results-champion-trial-interval.txt" using 1:2 title "w 0.05" with lines, \
     "./g15e200t100e99w10/results-champion-trial-interval.txt" using 1:2 title "w 0.10" with lines, \
     "./g15e200t100e99w20/results-champion-trial-interval.txt" using 1:2 title "w 0.20" with lines, \
     "./g15e200t100e99w25/results-champion-trial-interval.txt" using 1:2 title "w 0.25" with lines, \
     "./g15e200t100e99w50/results-champion-trial-interval.txt" using 1:2 title "w 0.50" with lines, \
     "./g15e200t100e99w100/results-champion-trial-interval.txt" using 1:2 title "w 1.00" with lines

set output "w-compare-population-interval.eps"
set xlabel "Episode"
set ylabel "Average Population Reward over 15 Generations"
plot "./g15e200t100e99w0/results-population-interval.txt" using 1:2 title "w 0.00" with lines, \
     "./g15e200t100e99w5/results-population-interval.txt" using 1:2 title "w 0.05" with lines, \
     "./g15e200t100e99w10/results-population-interval.txt" using 1:2 title "w 0.10" with lines, \
     "./g15e200t100e99w20/results-population-interval.txt" using 1:2 title "w 0.20" with lines, \
     "./g15e200t100e99w25/results-population-interval.txt" using 1:2 title "w 0.25" with lines, \
     "./g15e200t100e99w50/results-population-interval.txt" using 1:2 title "w 0.50" with lines, \
     "./g15e200t100e99w100/results-population-interval.txt" using 1:2 title "w 1.00" with lines
