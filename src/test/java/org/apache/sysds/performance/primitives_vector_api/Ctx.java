package org.apache.sysds.performance.primitives_vector_api;

public class Ctx {
    public int len;
    public double[] a, cInit, cScalar, cVector;
    public double bval;
  
    public double scalarRes, vectorRes;
    public double maxDiff;
    public boolean ok;
  
    void initDenseA() {
      a = new double[len];
      for (int i = 0; i < len; i++) a[i] = (i % 10) - 5;
    }
  
    void initDenseAandC() {
      initDenseA();
      cInit = new double[len];
      for (int i = 0; i < len; i++) cInit[i] = (i % 10) - 5;
      cScalar = java.util.Arrays.copyOf(cInit, len);
      cVector = java.util.Arrays.copyOf(cInit, len);
      bval = 1.234567;
    }
  
    void resetC() {
      if (cInit != null) {
        System.arraycopy(cInit, 0, cScalar, 0, len);
        System.arraycopy(cInit, 0, cVector, 0, len);
      }
    }
  }
  
