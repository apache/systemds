package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.ComplexDouble;
import org.apache.sysds.runtime.matrix.data.LibMatrixFourier;
import org.apache.sysds.runtime.matrix.data.LibMatrixFourier2;

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier2.fft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier2.transform;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;


public class Fourier2Test {

    @Test
    public void simpleTest() {
        // tested with numpy
        double[] in = {0, 18, -15, 3};
        ComplexDouble[] expected = new ComplexDouble[4];
        expected[0] = new ComplexDouble(6, 0);
        expected[1] =  new ComplexDouble(15, -15);
        expected[2] = new ComplexDouble(-36, 0);
        expected[3] = new ComplexDouble(15, 15);

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


}
