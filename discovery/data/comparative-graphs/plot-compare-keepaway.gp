set terminal postscript eps color

set xtics 200
#set grid
set yrange [:37]

set output "compare-keepaway.eps"
set xlabel "Episode"
set ylabel "Reward"
plot "./keepaway-g15e200t1e100w0/results-best-champion-interval.txt" using 1:2 title "Best Champion's Policy (20 Trials)" with lines, \
     "./keepaway-g15e200t1e100w0/results-champions-training-interval.txt" using 1:2 title "Champions over 15 Generations (Training)" with lines, \
     "./keepaway-t20e3000w0/results-standard-training-interval.txt" using 1:2 title "Handcoded Features Training (20 Trials)" with lines, \
     "./keepaway-t20e3000w0/results-standard-eval-interval.txt" using 1:2 title "Best Policy from Handcoded Features (20 Trials)" with lines

