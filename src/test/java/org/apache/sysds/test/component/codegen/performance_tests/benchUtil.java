package org.apache.sysds.test.component.codegen.performance_tests;


public class benchUtil {

    public static void warmup(Runnable r, int iters) {
        for (int i = 0; i < iters; i++) {
            r.run();
        }
    }

    /** returns ns per call */
    public static double measure(Runnable r, int iters) {
        long t0 = System.nanoTime();
        for (int i = 0; i < iters; i++) {
            r.run();
        }
        long t1 = System.nanoTime();
        return (t1 - t0) / (double) iters;
    }

    public static double checksum(double[] x) {
        double s = 0;
        for (double v : x) s += v;
        return s;
    }

    public static double maxAbsDiff(double[] a, double[] b) {
        double m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}

