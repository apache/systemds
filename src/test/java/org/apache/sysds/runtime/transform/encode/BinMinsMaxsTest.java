package org.apache.sysds.runtime.transform.encode;

import org.junit.Test;

import static org.junit.Assert.*;

public class BinMinsMaxsTest {
    @Test
    public void shouldCalculateSizeCorrectlyWhenBinMinsMaxsIsCreated(){
        // arrange
        double[] binMins = {1.0, 2.1, 3.1};
        double[] binMaxs = {2.0, 3.0, 4.0};

        // act
        BinMinsMaxs binMinsMaxs = BinMinsMaxs.create(binMins, binMaxs);

        // assert
        long actualSize = binMinsMaxs.getSize();
        long expectedSize = 72l;

        assertEquals(expectedSize, actualSize);
    }
}