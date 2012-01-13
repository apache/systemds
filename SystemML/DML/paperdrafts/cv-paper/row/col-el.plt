set terminal postscript enhanced color eps
set output "col-el.eps"
set xlabel "(D) numColumns, in thousands"
set xtics 100
set ylabel "time, s" offset 1.8
set key top left
set size 0.45,0.45
plot 'col-cv.data' u 1:3 w lp lc 2 lt 2 t "Join-Reblock", '' u 1:2 w lp lc 3 lt 3 t "HashMap"
