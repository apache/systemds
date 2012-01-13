set terminal postscript enhanced color eps
set output "row-el.eps"
set xlabel "(B) numRows, in millions"
set ylabel "time, s" offset 1.8
set key top left
set size 0.45,0.45
plot 'row-el.data' u 1:3 w lp lc 2 lt 2 t "Join-Reblock", '' u 1:2 w lp lc 3 lt 3 t "HashMap"
