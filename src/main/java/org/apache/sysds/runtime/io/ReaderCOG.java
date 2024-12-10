package org.apache.sysds.runtime.io;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.IOException;
import java.io.InputStream;

public class ReaderCOG extends MatrixReader{
    protected final FileFormatPropertiesCOG _props;

    public ReaderCOG(FileFormatPropertiesCOG props) {
        _props = props;
    }
    @Override
    public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {
        // Dummy data, the sum of this should be 100 (as is returned by the CSV test which
        // then has an assertion error of course)
        int rows = 10;
        int cols = 10;
        MatrixBlock dummyMatrix = new MatrixBlock(rows, cols, false);
        dummyMatrix.allocateDenseBlock();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dummyMatrix.set(i, j, 1.0);
            }
        }
        // Isn't printed when debugging (or running) the tests
        // I'll leave it for now
        System.out.println("We done something!");
        return dummyMatrix;
    }

    @Override
    public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {
        // Dummy data, the sum of this should be 100 (as is returned by the CSV test which
        // then has an assertion error of course)
        int rows = 10;
        int cols = 10;
        MatrixBlock dummyMatrix = new MatrixBlock(rows, cols, false);
        dummyMatrix.allocateDenseBlock();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dummyMatrix.set(i, j, 1.0);
            }
        }
        // Isn't printed when debugging (or running) the tests
        // I'll leave it for now
        System.out.println("We done something!");
        return dummyMatrix;
    }
}
