set terminal postscript eps color

set xtics 200
#set grid
set yrange [:0.6]

set output "compare-minisoccer.eps"
set xlabel "Episode"
set ylabel "Reward"
plot "./minisoccer-g15e200t1e100w0/results-best-champion-interval.txt" using 1:2 title "Best Champion's Policy (20 Trials)" with lines, \
     "./minisoccer-g15e200t1e100w0/results-champions-training-interval.txt" using 1:2 title "Champions over 15 Generations (Training)" with lines, \
     "./minisoccer-t20e3000w0/results-standard-training-interval.txt" using 1:2 title "Handcoded Features Training (20 Trials)" with lines, \
     "./minisoccer-t20e3000w0/results-standard-eval-interval.txt" using 1:2 title "Best Policy from Handcoded Features (20 Trials)" with lines

