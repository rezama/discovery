set terminal postscript eps color

set xtics 200
#set grid
set yrange [:0.6]

set output "compare-minisoccer-s.eps"
set xlabel "Episode"
set ylabel "Reward"
plot "./minisoccer-t20e3000w0/results-standard-training-interval.txt" using 1:2 title "Handcoded Features (20 Trials)" with lines, \
     "./minisoccer-t20e3000w0/results-standard-eval-interval.txt" using 1:2 title "Best Policy from Handcoded Features (20 Trials)" with lines

