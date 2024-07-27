package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

import java.util.Arrays;

public class LibMatrixIMGTransform {

    protected static final Log LOG = LogFactory.getLog(LibMatrixFourier.class.getName());

    /**
     * affine transformation matrix for calculated for a picture of original size and target dimensions
     * see: https://en.wikipedia.org/wiki/Affine_transformation
     *
     * #orig_w = as.scalar(dimMat[1,2])
     * #orig_h = as.scalar(dimMat[1,1])
     * #out_w = as.scalar(dimMat[2,2])
     * #out_h = as.scalar(dimMat[2,1])
     * T_inv = inv(transMat)
     *
     * ## coordinates of output pixel-centers linearized in row-major order
     * coords = matrix(1, rows=3, cols=out_w*out_h)
     * coords[1,] = t((seq(0, out_w*out_h-1) %% out_w) + 0.5)
     * coords[2,] = t((seq(0, out_w*out_h-1) %/% out_w) + 0.5)
     * # compute sampling pixel indices
     * coords = floor(T_inv %*% coords) + 1
     * inx = t(coords[1,])
     * iny = t(coords[2,])
     * # any out-of-range pixels, if present, correspond to an extra pixel with fill_value at the end of the input
     * index_vector = (orig_w *(iny-1) + inx) * ((0<inx) & (inx<=orig_w) & (0<iny) & (iny<=orig_h))
     * index_vector = t(index_vector)
     * xs = ((index_vector == 0)*(orig_w*orig_h +1)) + index_vector
     * #if(min(index_vector) == 0){
     * #  ys=cbind(img_in, matrix(fill_value,nrow(img_in), 1))
     * #}else{
     * #  ys = img_in
     * #}
     * ind= matrix(seq(1,ncol(xs),1),1,ncol(xs))
     * z = table(xs, ind)
     * zMat = transMat
     * isFillable = as.double(min(index_vector) == 0)
     */
    public static MatrixBlock[] transformationMatrix(MatrixBlock transMat, MatrixBlock dimMat, int threads) {
        //throw new NotImplementedException("If the code reaches here, we good boys");
        int orig_w = (int) dimMat.get(0,0);
        int orig_h = (int) dimMat.get(0,1);
        int out_w = (int) dimMat.get(1,0);
        int out_h = (int) dimMat.get(1,1);

        //calculate the inverse of the transformation matrix
        MatrixBlock t_Inv = LibCommonsMath.unaryOperations(transMat, "inverse");

        //greate the coords matrix
        MatrixBlock coords = new MatrixBlock(3, out_w*out_h, false);
        //coords.sparseToDense();
        //change values to coords[1, ] = t((seq(0, out_w * out_h - 1) %% out_w) + 0.5)
        //coords[2, ] = t((seq(0, out_w * out_h - 1) %/% out_w) + 0.5)
        double [] coords1 = new double[out_w*out_h];
        double [] coords2 = new double[out_w*out_h];
        double [] coords3 = new double[out_w*out_h];
        for(int i=0; i<out_w*out_h; i++) {
            coords1[i] = (i % out_w) + 0.5;
            coords2[i] = Math.floor((double) i / out_w) + 0.5;
            coords3[i] = 1.0;
        }
        coords.init(new double[][] {coords1, coords2, coords3}, 3, out_w*out_h);

        MatrixBlock coords_mul;

        op_mult = new BinaryOperator(Multiply.getMultiplyFnObject(), t_Inv, coords);


        //System.out.println(coords);
        //System.out.println(Arrays.toString(coords1));
        //System.out.println(Arrays.toString(coords2));
        MatrixBlock filledBlock = new MatrixBlock(21.3);
        //System.out.println("Input dimensions: w:" + orig_w + "; h" + orig_h);
        //System.out.println("Output dimensions: w:" + out_w + "; h" + out_h);
        //MatrixBlock transMatrix = new MatrixBlockDataOutput();
        return new MatrixBlock[] {transMat, filledBlock};
    }
}
