package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class LibMatrixIMGTransform {

    protected static final Log LOG = LogFactory.getLog(LibMatrixFourier.class.getName());

    public static MatrixBlock[] transformationMatrix(MatrixBlock transMat, MatrixBlock dimMat, int threads) {
        throw new NotImplementedException();
    }
}
