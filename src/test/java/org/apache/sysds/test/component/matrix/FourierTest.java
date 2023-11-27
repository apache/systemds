package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.ComplexDouble;
import org.apache.sysds.runtime.matrix.data.LibMatrixFourier;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft;

import org.junit.Test;

import java.util.Arrays;

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

        assertTrue(Arrays.equals(expected, res));
    }

}
