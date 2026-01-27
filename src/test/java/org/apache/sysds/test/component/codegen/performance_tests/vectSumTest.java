package org.apache.sysds.test.component.codegen.performance_tests;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;


public class vectSumTest {
    public static void main(String[] args) {
        int len = 1_000_000;
        double[] a = new double[len];
        for (int i = 0; i < len; i++)
            a[i] = (i % 10) - 5;
        float[] a_f = new float[len];
        for (int i = 0; i < len; i++)
            a_f[i] = (i % 10) - 5;

        // warm up
        for (int i = 0; i < 20_000; i++) {
            LibSpoofPrimitives.vectSum(a, 0, len);
            LibSpoofPrimitives.scalarvectSum(a, 0, len);
            LibSpoofPrimitives.vectSumFloat(a_f, 0, len);
            LibSpoofPrimitives.scalarvectSumFloat(a_f,0, len);
        }


        // measure
        long t2_0 = System.nanoTime();
        double s2 = 0;
        for (int i = 0; i < 2000; i++)
            s2 += LibSpoofPrimitives.scalarvectSum(a, 0, len);
        long t2_1 = System.nanoTime();

        System.out.println("Scalar Sum=" + s2);
        System.out.println("Time per call (ns): " + ((t2_1 - t2_0) / 2000.0));
        
        // measure
        long t1_0 = System.nanoTime();
        double s1 = 0;
        for (int i = 0; i < 2000; i++)
            s1 += LibSpoofPrimitives.vectSum(a, 0, len);
        long t1_1 = System.nanoTime();

        System.out.println("Vector Sum=" + s1);
        System.out.println("Time per call (ns): " + ((t1_1 - t1_0) / 2000.0));

        // measure
        long t3_0 = System.nanoTime();
        double s3 = 0;
        for (int i = 0; i < 2000; i++)
            s3 += LibSpoofPrimitives.vectSumFloat(a_f, 0, len);
        long t3_1 = System.nanoTime();

        System.out.println("Vector Float Sum=" + s3);
        System.out.println("Time per call (ns): " + ((t3_1 - t3_0) / 2000.0));


        // measure
        long t4_0 = System.nanoTime();
        double s4 = 0;
        for (int i = 0; i < 2000; i++)
            s4 += LibSpoofPrimitives.scalarvectSumFloat(a_f,0, len);
        long t4_1 = System.nanoTime();

        System.out.println("Scalar Float Sum=" + s4/2000);
        System.out.println("Time per call (ns): " + ((t4_1 - t4_0) / 2000.0));

    }
}
/* 
Scalar Sum=-1.0E9
Time per call (ns): 142774.5625
Vector Sum=-1.0E9
Time per call (ns): 468854.25
Vector Float Sum=-1.0E9
Time per call (ns): 274727.3545
*/
