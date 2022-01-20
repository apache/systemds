package org.apache.sysds.test.component.compress.colgroup;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Assert;
import org.junit.Test;

import java.util.EnumSet;

public class ColGroupDeltaDDCTest {

    @Test
    public void testDecompressToDenseBlockSingleColumn() {
        testDecompressToDenseBlock(new double[][] {{1,2,3,4,5}}, true);
    }

    @Test
    public void testDecompressToDenseBlockSingleColumnTransposed() {
        testDecompressToDenseBlock(new double[][] {{1},{2},{3},{4},{5}}, false);
    }

    @Test
    public void testDecompressToDenseBlockTwoColumns() {
        testDecompressToDenseBlock(new double[][] {{1,1},{2,1},{3,1},{4,1},{5,1}}, false);
    }

    @Test
    public void testDecompressToDenseBlockTwoColumnsTransposed() {
        testDecompressToDenseBlock(new double[][] {{1,2,3,4,5},{1,1,1,1,1}}, true);
    }

    public void testDecompressToDenseBlock(double[][] data, boolean isTransposed) {
        MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

        final int numCols = isTransposed ? mbt.getNumRows() : mbt.getNumColumns();
        final int numRows = isTransposed ? mbt.getNumColumns() : mbt.getNumRows();
        int[] colIndexes = new int[numCols];
        for(int x = 0; x < numCols; x++)
            colIndexes[x] = x;

        try {
            CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(1.0)
                    .setValidCompressions(EnumSet.of(AColGroup.CompressionType.DeltaDDC)).create();
            cs.transposed = isTransposed;
            ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, mbt, isTransposed, 8, false);

            EstimationFactors ef = CompressedSizeEstimator.estimateCompressedColGroupSize(ubm, colIndexes,
                    numRows, cs);
            CompressedSizeInfoColGroup cgi = new CompressedSizeInfoColGroup(colIndexes, ef, AColGroup.CompressionType.DeltaDDC);
            CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
            AColGroup cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

            // Decompress to dense block
            MatrixBlock ret = new MatrixBlock(numRows, numCols, false);
            ret.allocateDenseBlock();
            cg.decompressToDenseBlock(ret.getDenseBlock(), 0, numRows);

            MatrixBlock expected = DataConverter.convertToMatrixBlock(data);
            if(isTransposed)
                LibMatrixReorg.transposeInPlace(expected, 1);
            Assert.assertArrayEquals(expected.getDenseBlockValues(), ret.getDenseBlockValues(), 0.01);

        }
        catch(Exception e) {
            e.printStackTrace();
            throw new DMLRuntimeException("Failed construction : " + this.getClass().getSimpleName());
        }
    }

}
