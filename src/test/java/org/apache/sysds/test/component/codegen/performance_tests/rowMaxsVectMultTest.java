package org.apache.sysds.test.component.codegen.performance_tests;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;


public class rowMaxsVectMultTest {
    public static void main(String[] args) {
        int len = 1_000_000;
        double[] a = new double[len];
        for (int i = 0; i < len; i++)
            a[i] = (i % 10) - 5;
        double[] b = new double[len];
        for (int i = 0; i < len; i++)
            b[i] = (i % 10) - 5;

        float[] a_f = new float[len];
        for (int i = 0; i < len; i++)
            a_f[i] = (i % 10) - 5;
        float[] b_f = new float[len];
        for (int i = 0; i < len; i++)
            b_f[i] = (i % 10) - 5;



        // warm up
        for (int i = 0; i < 20_000; i++) {
            LibSpoofPrimitives.rowMaxsVectMult(a, b, 0,0,len);
            LibSpoofPrimitives.scalarrowMaxsVectMult(a, b,0,0, len);
            LibSpoofPrimitives.rowMaxsVectMultFloat(a_f, b_f,0,0, len);
            LibSpoofPrimitives.scalarrowMaxsVectMultFloat(a_f, b_f,0,0, len);
            LibSpoofPrimitives.rowMaxsVectMultVec2Acc(a, b,0,0, len);
        }

        // measure
        long t2_0 = System.nanoTime();
        double s2 = 0;
        for (int i = 0; i < 2000; i++)
            s2 += LibSpoofPrimitives.rowMaxsVectMult(a, b, 0,0,len);
        long t2_1 = System.nanoTime();

        System.out.println("Vector MaxVal=" + s2/2000);
        System.out.println("Time per call (ns): " + ((t2_1 - t2_0) / 2000.0));
        
        // measure
        long t1_0 = System.nanoTime();
        double s1 = 0;
        for (int i = 0; i < 2000; i++)
            s1 += LibSpoofPrimitives.scalarrowMaxsVectMult(a, b,0,0, len);
        long t1_1 = System.nanoTime();

        System.out.println("Scalar MaxVal Sum=" + s1/2000);
        System.out.println("Time per call (ns): " + ((t1_1 - t1_0) / 2000.0));


        // measure
        long t3_0 = System.nanoTime();
        double s3 = 0;
        for (int i = 0; i < 2000; i++)
            s3 += LibSpoofPrimitives.rowMaxsVectMultFloat(a_f, b_f,0,0, len);
        long t3_1 = System.nanoTime();

        System.out.println("Vector Float MaxVal=" + s3/2000);
        System.out.println("Time per call (ns): " + ((t3_1 - t3_0) / 2000.0));

        // measure
        long t4_0 = System.nanoTime();
        double s4 = 0;
        for (int i = 0; i < 2000; i++)
            s4 += LibSpoofPrimitives.scalarrowMaxsVectMultFloat(a_f, b_f,0,0, len);
        long t4_1 = System.nanoTime();

        System.out.println("Scalar Float MaxVal=" + s4/2000);
        System.out.println("Time per call (ns): " + ((t4_1 - t4_0) / 2000.0));

        // measure
        long t5_0 = System.nanoTime();
        double s5 = 0;
        for (int i = 0; i < 2000; i++)
            s5 += LibSpoofPrimitives.rowMaxsVectMultVec2Acc(a, b,0,0, len);
        long t5_1 = System.nanoTime();

        System.out.println("Vector 2acc MaxVal=" + s5/2000);
        System.out.println("Time per call (ns): " + ((t5_1 - t5_0) / 2000.0));

    

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
