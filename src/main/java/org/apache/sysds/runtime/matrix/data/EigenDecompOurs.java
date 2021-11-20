package org.apache.sysds.runtime.matrix.data;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.DataConverter;

import java.io.Console;
import java.util.Arrays;

// TODO: EigenDecompostition entirely in SystemDS matrix-operators
//		 (find out what the computeEigenRegularized function does and when it is needed)

// NOTES: what does org.apache.commons.math3.linear.EigenDecomposition use?
//			- only for real matrices (symmetric and non-symmetric)
//			- we only use the real valued eigenvalues
//			- This implementation is based on the paper "The Implicit QL Algorithm" (1971)
//			- similar to JAMA implementation
//
//	      Apache common source
//			- https://gitbox.apache.org/repos/asf?p=commons-math.git
//
//		  Symmetric:
//			- tred2 (p.217) -> tql2 (p.244)	(list of procedures p. 192)
//		  Non-symmetric:
//			- othes (p.349) (or dirhes, elmhes) -> hqr2 (p. 383)
//
//        Make sure to return the EVec and EVal sorted by values of Eval descending order

public class EigenDecompOurs {
    private double[][] m;
    private double[][] tred2_z;
    private double[] tred2_e;
    private double[] tred2_d;

    public EigenDecompOurs(MatrixBlock in) {
        if ( in.getNumRows() != in.getNumColumns() ) {
            throw new DMLRuntimeException("Eigen Decomposition can only be done on a square matrix. "
                    + "Input matrix is rectangular (rows=" + in.getNumRows() + ", cols="+ in.getNumColumns() +")");
        }
        this.m= DataConverter.convertToArray2DRowRealMatrix(in).getData();
        if(isSym(this.m)) {
            tred2(this.m);
            printMatrix(this.tred2_z);
            tql2();
        }

    }

    private boolean isSym(double[][] m) {

        for(int i = 0; i < m.length; i++) {
            for(int j = 0; j < m[i].length; j++) {
                if (Double.compare(m[i][j], m[j][i]) != 0)
                    return false;
            }
        }
        return true;
    }

    // "Handbook for Automatic Computation" (1971)
    // tred2 on page 217
    private void tred2(double[][] m) {
        double tol = 1e-7;

        int n = m.length;
        double[][] z = new double[n][n];
        double[] e = new double[n];
        double[] d = new double[n];

        for(int i = 0; i < n; i++) {
            e[i] = 0;
            d[i] = 0;
            for(int j = 0; j < n; j++) {
                if (j < i)
                    z[i][j] = m[i][j];
                else
                    z[i][j] = 0.0;
            }
        }
        for(int i = n-1; i > 0; i--) {
            int l = i-2;
            double f = z[i][i-1];
            double g = 0.0;
            for(int k = 0; k < l; k++) {
                g += (z[i][k] * z[i][k]);
            }
            double h = g + f * f;
            if (g <= tol) {
                e[i] = f;
                d[i] = 0.0;
                continue;
            }
            l = l+1;
            g = f >= 0 ? (-1 * Math.sqrt(h)) : Math.sqrt(h);
            e[i] = g;
            h = h - f * g;
            z[i][i-1] = f - g;
            f = 0;
            for(int j = 0; j < l; j++) {
                z[j][i] = z[i][j] / h;
                g = 0;
                for(int k = 0; k < j; k++) {
                    g += (z[j][k] * z[i][k]);
                }
                for(int k = j; k < l; k++) {
                    g += (z[k][j] * z[i][k]);
                }
                e[j] = g / h;
                f += (g * z[j][i]);
            }
            double hh = f / (h + h);
            for(int j = 0; j < l; j++) {
                f = z[i][j];
                e[j] -= (hh * f);
                g = e[j];
                for(int k = 0; k < j; k++) {
                    z[j][k] -= (f * e[k] + g * z[i][k]);
                }
            }
            d[i] = h;
        }
        d[0] = 0.0;
        e[0] = 0.0;
        for(int i = 0; i < n; i++) {
            int l = i - 1;
            if (d[i] != 0.0) {
                for(int j = 0; j < l; j++) {
                    double g = 0.0;
                    for(int k = 0; k < l; k++) {
                        g += (z[i][k] * z[k][j]);
                    }
                    for(int k = 0; k < l; k++) {
                        z[k][j] -= (g * z[k][i]);
                    }
                }
            }
            d[i] = z[i][i];
            z[i][i] = 1.0;
            for(int j = 0; j < l; j++) {
                z[i][j] = 0.0;
                z[j][i] = 0.0;
            }
        }
        this.tred2_d = d;
        this.tred2_e = e;
        this.tred2_z = z;
    }

    private MatrixBlock tql2() {
        return null;
    }

    private void printMatrix(double[][] m) {
        for(double[] a : m){
            System.out.println(Arrays.toString(a));
        }
    }

    public MatrixBlock getV() {
        return null;
    }

    public MatrixBlock getRealEigenvalues() {
        return null;
    }
}

