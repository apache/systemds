package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.ComplexDouble;
import org.apache.sysds.runtime.matrix.data.LibMatrixFourier;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft2d;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

public class FourierTest {

    @Test
    public void simpleTest() {
        double[] in = {0, 18, -15, 3};
        ComplexDouble[] expected = new ComplexDouble[4];
        expected[0] = new ComplexDouble(6, 0);
        expected[1] =  new ComplexDouble(15, 15);
        expected[2] = new ComplexDouble(-36, 0);
        expected[3] = new ComplexDouble(15, -15);

        ComplexDouble[] res = fft(in);
        for(ComplexDouble elem : res){
            System.out.println(elem);
        }

        assertArrayEquals(expected, res);
    }

    @Test
    public void notPowerOfTwoTest() {
        double[] in = {1, 2, 3, 4, 5};

        // see https://de.mathworks.com/help/matlab/ref/ifft.html
        ComplexDouble[] expected = new ComplexDouble[5];
        expected[0] = new ComplexDouble(15,0);
        expected[1] = new ComplexDouble(-2.5000,3.4410);
        expected[2] = new ComplexDouble(-2.5000,0.8123);
        expected[3] = new ComplexDouble(-2.5000, - 0.8123);
        expected[4] = new ComplexDouble(-2.5000, - 3.4410);

        ComplexDouble[] res = fft(in);
        for(ComplexDouble elem : res){
            System.out.println(elem);
        }
        assertArrayEquals(expected, res);
    }

    @Test
    public void simple2dTest() {
        // tested with matlab
        double[][] in = {{0, 18}, {-15, 3}};
        ComplexDouble[][] expected = new ComplexDouble[2][2];
        expected[0][0] = new ComplexDouble(6, 0);
        expected[0][1] = new ComplexDouble(-36, 0);
        expected[1][0] =  new ComplexDouble(30, 0);
        expected[1][1] =  new ComplexDouble(0, 0);

        ComplexDouble[][] res = fft2d(in);
        for(ComplexDouble[] row : res){
            for(ComplexDouble elem : row){
                System.out.println(elem);
            }
        }

        assertArrayEquals(expected, res);
    }

    @Test
    public void simple2dTest2() {
        double[][] in = {{0, 18, -15, 3}, {0, 18, -15, 3}};
        ComplexDouble[][] expected = new ComplexDouble[2][4];
        expected[0][0] = new ComplexDouble(12, 0);
        expected[0][1] = new ComplexDouble(30, -30);
        expected[0][2] = new ComplexDouble(-72, 0);
        expected[0][3] = new ComplexDouble(30, 30);
        expected[1][0] =  new ComplexDouble(0, 0);
        expected[1][1] =  new ComplexDouble(0, 0);
        expected[1][2] = new ComplexDouble(0, 0);
        expected[1][3] = new ComplexDouble(0, 0);

        ComplexDouble[][] res = fft2d(in);
        for(ComplexDouble[] row : res){
            for(ComplexDouble elem : row){
                System.out.println(elem);
            }
        }

        assertArrayEquals(expected, res);
    }

}
