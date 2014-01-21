set terminal postscript eps color

set xtics 200
#set grid
set yrange [:470]

set output "compare-knightjoust-s.eps"
set xlabel "Episode"
set ylabel "Reward"
plot "./knightjoust-t20e3000w0/results-standard-training-interval.txt" using 1:2 title "Handcoded Features (20 Trials)" with lines, \
     "./knightjoust-t20e3000w0/results-standard-eval-interval.txt" using 1:2 title "Best Policy from Handcoded Features (20 Trials)" with lines

