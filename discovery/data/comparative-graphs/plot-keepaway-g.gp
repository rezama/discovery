set terminal postscript eps color

set xtics 200
#set grid
set yrange [:37]

set output "compare-keepaway-g.eps"
set xlabel "Episode"
set ylabel "Reward"
plot "./keepaway-g15e200t1e100w0/results-population-interval.txt" using 1:2 title "Average Population's Reward (20 Trials)" with lines, \
     "./keepaway-g15e200t1e100w0/results-champions-training-interval.txt" using 1:2 title "Generation Champions' Reward (20 Trials)" with lines

