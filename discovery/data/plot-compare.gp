set terminal postscript eps color

set xtics 200
#set grid
#set yrange [:520]

set output "compare-knightjoust.eps"
set xlabel "Episode"
set ylabel "Reward"
plot "./knightjoust-g15e200t1e100w0/results-champions-training-interval.txt" using 1:2 title "Champions Training" with lines, \
     "./knightjoust-g15e200t1e100w0/results-best-champion-interval.txt" using 1:2 title "Best Champion's Policy 20 Trials" with lines, \
     "./knightjoust-t20e3000w0/results-standard-training-interval.txt" using 1:2 title "Handcoded Features Training 20 Trials" with lines, \
     "./knightjoust-t20e3000w0/results-standard-eval-interval.txt" using 1:2 title "Handcoded Features Policy 20 Trials" with lines

