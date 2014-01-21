set terminal postscript eps color

set xtics 200
#set grid
set yrange [:470]

set output "compare-knightjoust.eps"
set xlabel "Episode"
set ylabel "Reward"
plot "./knightjoust-g15e200t1e100w0/results-best-champion-interval.txt" using 1:2 title "Best Champion (20 Trials)" with lines, \
     "./knightjoust-g15e200t1e100w0/results-champions-training-interval.txt" using 1:2 title "Generation Champions over 15 Generations (1 Trial)" with lines, \
     "./knightjoust-t20e3000w0/results-standard-training-interval.txt" using 1:2 title "Handcoded Features (20 Trials)" with lines, \
     "./knightjoust-t20e3000w0/results-standard-eval-interval.txt" using 1:2 title "Best Policy from Handcoded Features (20 Trials)" with lines

