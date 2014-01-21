set terminal postscript eps color

set xtics 200
#set grid
set yrange [:37]

set output "compare-keepaway-s.eps"
set xlabel "Episode"
set ylabel "Reward"
plot "./keepaway-t20e3000w0/results-standard-training-interval.txt" using 1:2 title "Handcoded Features (20 Trials)" with lines, \
     "./keepaway-t20e3000w0/results-standard-eval-interval.txt" using 1:2 title "Best Policy from Handcoded Features (20 Trials)" with lines

