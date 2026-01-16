/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.*;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class ColGroupDDCTest {

    protected static final Log LOG = LogFactory.getLog(ColGroupDDCTest.class.getName());

    @Test
    public void testConvertToDDCLZWBasic() {
        // TODO: neue Methode zum Vergleich
        IColIndex colIndexes = ColIndexFactory.create(2);
        double[] dictValues = new double[]{10.0, 20.0, 11.0, 21.0, 12.0, 22.0};
        Dictionary dict = Dictionary.create(dictValues);

        int[] src = new int[]{
                // repeating base pattern
                2, 0, 2, 1, 0, 2, 1, 0, 2,
                2, 0, 2, 1, 0, 2, 1, 0, 2,

                // variation / shifted pattern
                1, 0, 1, 2, 0, 1, 2, 0, 1,
                1, 0, 1, 2, 0, 1, 2, 0, 1,

                // longer runs (good for phrase growth)
                2, 2, 2, 2, 2,
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,

                // mixed noise
                2, 1, 0, 2, 1, 0, 2, 1, 0,
                0, 2, 1, 0, 2, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1,

                // repeating tail (tests dictionary reuse)
                2, 0, 2, 1, 0, 2, 1, 0, 2,
                2, 0, 2, 1, 0, 2, 1, 0, 2, 0, 0, 0, 0, 0, 1
        };

        final int nRows = src.length;
        final int nUnique = 3;
        AMapToData data = MapToFactory.create(nRows, nUnique);
        for (int i = 0; i < nRows; i++)
            data.set(i, src[i]);

        ColGroupDDC ddc = (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
        AColGroup result = ddc.convertToDDCLZW();

        assertNotNull(result);
        assertTrue(result instanceof ColGroupDDCLZW);

        ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) result;
        AColGroup ddclzwDecompressed = ddclzw.convertToDDC();

        assertNotNull(ddclzwDecompressed);
        assertTrue(ddclzwDecompressed instanceof ColGroupDDC);

        ColGroupDDC ddc2 = (ColGroupDDC) ddclzwDecompressed;

        AMapToData d1 = ddc.getMapToData();
        AMapToData d2 = ddc2.getMapToData();

        assertEquals(d1.size(), d2.size());
        assertEquals(d1.getUnique(), d2.getUnique());
        for (int i = 0; i < d1.size(); i++)
            assertEquals("mapping mismatch at row " + i, d1.getIndex(i), d2.getIndex(i));

        assertEquals(ddc.getColIndices(), ddc2.getColIndices());

        // Testen der Teildekompression:
        // Index entspricht der Anzahl der Zeichen, die dekodiert werden sollen (0 bis Index-1)
        int index = 10;
        ColGroupDDC ddcIndex = (ColGroupDDC) ddclzw.convertToDDC(index);

        AMapToData d3 = ddcIndex.getMapToData();
        assertEquals(index, d3.size());
        assertEquals(ddc.getColIndices(), ddcIndex.getColIndices());

        for(int i = 0; i < index; i++){
            assertEquals(d1.getIndex(i), d3.getIndex(i));
        }

        // Testen von SliceRows
        int low = 3;
        int high = 10;
        AColGroup slice = ddclzw.sliceRows(low, high);
        if(slice instanceof ColGroupDDCLZW ddclzwslice){
            ColGroupDDC ddcSlice = (ColGroupDDC) ddclzwslice.convertToDDC();
            ColGroupDDC ddcSlice2 = (ColGroupDDC) ddc.sliceRows(low, high);

            AMapToData d4 = ddcSlice.getMapToData();
            AMapToData d5 = ddcSlice2.getMapToData();

            assertEquals(d5.size(), d4.size());
            assertEquals(d5.getUnique(), d4.getUnique());

            for (int i = 0; i < d4.size(); i++)
                assertEquals("mapping mismatch at row " + i, d4.getIndex(i), d5.getIndex(i));
        }

    }

    @Test
    public void testConvertToDeltaDDCBasic() {
        IColIndex colIndexes = ColIndexFactory.create(2);
        double[] dictValues = new double[]{10.0, 20.0, 11.0, 21.0, 12.0, 22.0};
        Dictionary dict = Dictionary.create(dictValues);
        AMapToData data = MapToFactory.create(3, 3);
        data.set(0, 0);
        data.set(1, 1);
        data.set(2, 2);

        ColGroupDDC ddc = (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
        AColGroup result = ddc.convertToDeltaDDC();

        assertNotNull(result);
        assertTrue(result instanceof ColGroupDeltaDDC);
        ColGroupDeltaDDC deltaDDC = (ColGroupDeltaDDC) result;

        MatrixBlock mb = new MatrixBlock(3, 2, false);
        mb.allocateDenseBlock();
        deltaDDC.decompressToDenseBlock(mb.getDenseBlock(), 0, 3);

        assertEquals(10.0, mb.get(0, 0), 0.0);
        assertEquals(20.0, mb.get(0, 1), 0.0);
        assertEquals(11.0, mb.get(1, 0), 0.0);
        assertEquals(21.0, mb.get(1, 1), 0.0);
        assertEquals(12.0, mb.get(2, 0), 0.0);
        assertEquals(22.0, mb.get(2, 1), 0.0);
    }

    @Test
    public void testConvertToDeltaDDCSingleColumn() {
        IColIndex colIndexes = ColIndexFactory.create(1);
        double[] dictValues = new double[]{1.0, 2.0, 3.0, 4.0, 5.0};
        Dictionary dict = Dictionary.create(dictValues);
        AMapToData data = MapToFactory.create(5, 5);
        for (int i = 0; i < 5; i++)
            data.set(i, i);

        ColGroupDDC ddc = (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
        AColGroup result = ddc.convertToDeltaDDC();

        assertNotNull(result);
        assertTrue(result instanceof ColGroupDeltaDDC);
        ColGroupDeltaDDC deltaDDC = (ColGroupDeltaDDC) result;

        MatrixBlock mb = new MatrixBlock(5, 1, false);
        mb.allocateDenseBlock();
        deltaDDC.decompressToDenseBlock(mb.getDenseBlock(), 0, 5);

        assertEquals(1.0, mb.get(0, 0), 0.0);
        assertEquals(2.0, mb.get(1, 0), 0.0);
        assertEquals(3.0, mb.get(2, 0), 0.0);
        assertEquals(4.0, mb.get(3, 0), 0.0);
        assertEquals(5.0, mb.get(4, 0), 0.0);
    }

    @Test
    public void testConvertToDeltaDDCWithRepeatedValues() {
        IColIndex colIndexes = ColIndexFactory.create(2);
        double[] dictValues = new double[]{10.0, 20.0, 10.0, 20.0, 10.0, 20.0};
        Dictionary dict = Dictionary.create(dictValues);
        AMapToData data = MapToFactory.create(3, 3);
        data.set(0, 0);
        data.set(1, 1);
        data.set(2, 2);

        ColGroupDDC ddc = (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
        AColGroup result = ddc.convertToDeltaDDC();

        assertNotNull(result);
        assertTrue(result instanceof ColGroupDeltaDDC);
        ColGroupDeltaDDC deltaDDC = (ColGroupDeltaDDC) result;

        MatrixBlock mb = new MatrixBlock(3, 2, false);
        mb.allocateDenseBlock();
        deltaDDC.decompressToDenseBlock(mb.getDenseBlock(), 0, 3);

        assertEquals(10.0, mb.get(0, 0), 0.0);
        assertEquals(20.0, mb.get(0, 1), 0.0);
        assertEquals(10.0, mb.get(1, 0), 0.0);
        assertEquals(20.0, mb.get(1, 1), 0.0);
        assertEquals(10.0, mb.get(2, 0), 0.0);
        assertEquals(20.0, mb.get(2, 1), 0.0);
    }

    @Test
    public void testConvertToDeltaDDCWithNegativeDeltas() {
        IColIndex colIndexes = ColIndexFactory.create(2);
        double[] dictValues = new double[]{10.0, 20.0, 8.0, 15.0, 12.0, 25.0};
        Dictionary dict = Dictionary.create(dictValues);
        AMapToData data = MapToFactory.create(3, 3);
        data.set(0, 0);
        data.set(1, 1);
        data.set(2, 2);

        ColGroupDDC ddc = (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
        AColGroup result = ddc.convertToDeltaDDC();

        assertNotNull(result);
        assertTrue(result instanceof ColGroupDeltaDDC);
        ColGroupDeltaDDC deltaDDC = (ColGroupDeltaDDC) result;

        MatrixBlock mb = new MatrixBlock(3, 2, false);
        mb.allocateDenseBlock();
        deltaDDC.decompressToDenseBlock(mb.getDenseBlock(), 0, 3);

        assertEquals(10.0, mb.get(0, 0), 0.0);
        assertEquals(20.0, mb.get(0, 1), 0.0);
        assertEquals(8.0, mb.get(1, 0), 0.0);
        assertEquals(15.0, mb.get(1, 1), 0.0);
        assertEquals(12.0, mb.get(2, 0), 0.0);
        assertEquals(25.0, mb.get(2, 1), 0.0);
    }

    @Test
    public void testConvertToDeltaDDCWithZeroDeltas() {
        IColIndex colIndexes = ColIndexFactory.create(2);
        double[] dictValues = new double[]{5.0, 0.0, 5.0, 0.0, 0.0, 5.0};
        Dictionary dict = Dictionary.create(dictValues);
        AMapToData data = MapToFactory.create(3, 3);
        data.set(0, 0);
        data.set(1, 1);
        data.set(2, 2);

        ColGroupDDC ddc = (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
        AColGroup result = ddc.convertToDeltaDDC();

        assertNotNull(result);
        assertTrue(result instanceof ColGroupDeltaDDC);
        ColGroupDeltaDDC deltaDDC = (ColGroupDeltaDDC) result;

        MatrixBlock mb = new MatrixBlock(3, 2, false);
        mb.allocateDenseBlock();
        deltaDDC.decompressToDenseBlock(mb.getDenseBlock(), 0, 3);

        assertEquals(5.0, mb.get(0, 0), 0.0);
        assertEquals(0.0, mb.get(0, 1), 0.0);
        assertEquals(5.0, mb.get(1, 0), 0.0);
        assertEquals(0.0, mb.get(1, 1), 0.0);
        assertEquals(0.0, mb.get(2, 0), 0.0);
        assertEquals(5.0, mb.get(2, 1), 0.0);
    }

    @Test
    public void testConvertToDeltaDDCMultipleUniqueDeltas() {
        IColIndex colIndexes = ColIndexFactory.create(2);
        double[] dictValues = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        Dictionary dict = Dictionary.create(dictValues);
        AMapToData data = MapToFactory.create(4, 4);
        for (int i = 0; i < 4; i++)
            data.set(i, i);

        ColGroupDDC ddc = (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
        AColGroup result = ddc.convertToDeltaDDC();

        assertNotNull(result);
        assertTrue(result instanceof ColGroupDeltaDDC);
        ColGroupDeltaDDC deltaDDC = (ColGroupDeltaDDC) result;

        MatrixBlock mb = new MatrixBlock(4, 2, false);
        mb.allocateDenseBlock();
        deltaDDC.decompressToDenseBlock(mb.getDenseBlock(), 0, 4);

        assertEquals(1.0, mb.get(0, 0), 0.0);
        assertEquals(2.0, mb.get(0, 1), 0.0);
        assertEquals(3.0, mb.get(1, 0), 0.0);
        assertEquals(4.0, mb.get(1, 1), 0.0);
        assertEquals(5.0, mb.get(2, 0), 0.0);
        assertEquals(6.0, mb.get(2, 1), 0.0);
        assertEquals(7.0, mb.get(3, 0), 0.0);
        assertEquals(8.0, mb.get(3, 1), 0.0);
    }
}

