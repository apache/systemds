
package org.apache.sysds.test.component.codegen.performance_tests;
import java.util.Arrays;

import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;


public class vectDivAddTest {
    public static void main(String[] args) {
        //final int len = 32_768;
        final int len = 262_144;
        //final int len = 1_000_000;

        final double[] a = new double[len];
        final double[] cInit = new double[len];

        for (int i = 0; i < len; i++) {
            a[i] = (i % 10) - 5;
            cInit[i] = (i % 10) - 5;
        }

        final double bval = 1.234567; // NOT 1.0

        double[] cScalar = Arrays.copyOf(cInit, len);
        double[] cVector = Arrays.copyOf(cInit, len);
        double[] cVectorPureDiv = Arrays.copyOf(cInit, len);

        // Warm up scalar only
        for (int i = 0; i < 200; i++) {
            LibSpoofPrimitives.scalarvectDivAdd(a, bval, cScalar, 0, 0, len);
        }

        // Warm up vector only
        for (int i = 0; i < 200; i++) {
            LibSpoofPrimitives.vectDivAdd(a, bval, cVector, 0, 0, len);
        }

        // Warm up pure div vector only
        for (int i = 0; i < 200; i++) {
            LibSpoofPrimitives.pureDivvectDivAdd(a, bval, cVectorPureDiv, 0, 0, len);
        }

        // Reset for measurement
        cScalar = Arrays.copyOf(cInit, len);

        // Measure scalar
        long t0 = System.nanoTime();
        for (int i = 0; i < 2000; i++) {
            LibSpoofPrimitives.scalarvectDivAdd(a, bval, cScalar, 0, 0, len);
        }
        long t1 = System.nanoTime();

        // Reset for measurement
        cVector = Arrays.copyOf(cInit, len);

        // Measure vector
        long t2 = System.nanoTime();
        for (int i = 0; i < 2000; i++) {
            LibSpoofPrimitives.vectDivAdd(a, bval, cVector, 0, 0, len);
        }
        long t3 = System.nanoTime();

        // Compare correctness
        double maxDiff = 0;
        double sumScalar = 0, sumVector = 0;
        for (int i = 0; i < len; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(cScalar[i] - cVector[i]));
            sumScalar += cScalar[i];
            sumVector += cVector[i];
        }


         // Reset for measurement
         cVectorPureDiv = Arrays.copyOf(cInit, len);

         // Measure vector
         long t4 = System.nanoTime();
         for (int i = 0; i < 2000; i++) {
             LibSpoofPrimitives.pureDivvectDivAdd(a, bval, cVectorPureDiv, 0, 0, len);
         }
         long t5 = System.nanoTime();
 
         // Compare correctness

         double sum_prev = sumScalar + sumVector;
         double sum_Vector_pure_div = 0;
         for (int i = 0; i < len; i++) {
             maxDiff = Math.max(maxDiff, Math.abs(sumScalar - cVectorPureDiv[i]));
             sum_Vector_pure_div += cVectorPureDiv[i];
         }

        System.out.println("Scalar time per call (ns): " + ((t1 - t0) / 2000.0));
        System.out.println("Vector time per call (ns): " + ((t3 - t2) / 2000.0));
        System.out.println("pure vector div time per call (ns): " + ((t5 - t4) / 2000.0));
        System.out.println("maxDiff: " + maxDiff);
        System.out.println("checksum scalar: " + sumScalar);
        System.out.println("checksum vector: " + sumVector);
        System.out.println("checksum pure vector div : " + sum_Vector_pure_div);
    }
}
