set terminal postscript enhanced color eps
set output "sparsity.eps"
set xlabel "Sparsity"
set ylabel "time, s" offset 1.8
set xtics 0.02
set key top left
set size 0.45,0.45
plot 'sparsity.data' u 1:4 w lp lc 1 lt 1 t "Reblock-Based", '' u 1:3 w lp lc 2 lt 2 t "Join-Reblock", '' u 1:2 w lp lc 3 lt 3 t "HashMap"
