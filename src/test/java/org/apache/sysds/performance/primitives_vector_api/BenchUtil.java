package org.apache.sysds.performance.primitives_vector_api;


public class BenchUtil {
    public static volatile double blackhole;
  
    public static void warmup(Runnable r,int iters ) {
      for (int i = 0; i < iters; i++) r.run();
    }
  
    public static double measure(Runnable r,int iters) {
      System.gc();
      long t0 = System.nanoTime();
      for (int i = 0; i < iters; i++) r.run();
      long t1 = System.nanoTime();
      return (t1 - t0) / (double) iters;
    }

    // ---- args helpers ----
    public static int argInt(String[] args, String key, int def) {
        for (int i = 0; i < args.length - 1; i++)
        if (args[i].equals(key))
            return Integer.parseInt(args[i + 1]);
        return def;
    }

    public static String argStr(String[] args, String key, String def) {
        for (int i = 0; i < args.length - 1; i++)
        if (args[i].equals(key))
            return args[i + 1];
        return def;
    }
  
    public static double maxAbsDiff(double[] a, double[] b) {
      double m = 0;
      for (int i = 0; i < a.length; i++)
        m = Math.max(m, Math.abs(a[i] - b[i]));
      return m;
    }
  
    public static void printScalarDouble(String name,
        double nsScalar, double nsVector,
        double scalarRes, double vectorRes,
        boolean ok) {
  
      double speedup = nsScalar / nsVector;
      System.out.printf("%s | scalar %.1f ns | vector %.1f ns | speedup %.3fx | " +
                        "s=%.6g v=%.6g | %s%n",
          name, nsScalar, nsVector, speedup, scalarRes, vectorRes, ok ? "OK" : "FAIL");
    }
  
    public static void printArrayDiff(String name,
        double nsScalar, double nsVector,
        double maxDiff,
        boolean ok) {
  
      double speedup = nsScalar / nsVector;
      System.out.printf("%s | scalar %.1f ns | vector %.1f ns | speedup %.3fx | " +
                        "maxDiff=%.6g | %s%n",
          name, nsScalar, nsVector, speedup, maxDiff, ok ? "OK" : "FAIL");
    }
  }
  