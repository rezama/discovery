for d in `ls -d */`
do
    echo $d
    cd $d
    gnuplot ../../../src/plot/plot-ev.gp
    cd ..
done
