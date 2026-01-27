
package org.apache.sysds.test.component.codegen.performance_tests;
import java.util.Arrays;

import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;


public class vectEqualWriteTest {
    public static void main(String[] args) {
        //final int len = 32_768;
        //final int len = 262_144;
        final int len = 1_000_000;
        //final int len = 1_000_000;

        final double[] aInit = new double[len];

        for (int i = 0; i < len; i++) {
            aInit[i] = (i % 10) - 5;
        }

        final double bval = 1.234567; // NOT 1.0

        double[] aScalar = Arrays.copyOf(aInit, len);
        double[] aVector = Arrays.copyOf(aInit, len);

        // Warm up scalar only
        for (int i = 0; i < 200; i++) {
            LibSpoofPrimitives.scalarvectEqualWrite(aScalar, bval, 0,len);
        }

        // Warm up vector only
        for (int i = 0; i < 200; i++) {
            LibSpoofPrimitives.vectEqualWrite(aVector, bval, 0,len);
        }

        // Reset for measurement
        aScalar = Arrays.copyOf(aInit, len);

        // Measure scalar
        long t0 = System.nanoTime();
        for (int i = 0; i < 2000; i++) {
            LibSpoofPrimitives.scalarvectEqualWrite(aScalar, bval, 0,len);
        }
        long t1 = System.nanoTime();
        System.out.println("Scalar");
        System.out.println("Time per call (ns): " + ((t1- t0) / 2000.0));
        

        // Reset for measurement
        aVector = Arrays.copyOf(aInit, len);

        // Measure vector
        long t2 = System.nanoTime();
        for (int i = 0; i < 2000; i++) {
            LibSpoofPrimitives.vectEqualWrite(aVector, bval, 0,len);
        }
        long t3 = System.nanoTime();
        System.out.println("Vector");
        System.out.println("Time per call (ns): " + ((t3- t2) / 2000.0));
    }
}
