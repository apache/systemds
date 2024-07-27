package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

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

        assert t_Inv != null;
        AggregateBinaryOperator op_mul_agg = InstructionUtils.getMatMultOperator(threads);
        UnaryOperator op_floor = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.FLOOR));
        BinaryOperator op_plus = InstructionUtils.parseExtendedBinaryOperator("+");
        coords_mul = t_Inv.aggregateBinaryOperations(t_Inv, coords, op_mul_agg);
        coords_mul = coords_mul.unaryOperations(op_floor);
        coords_mul = coords_mul.binaryOperationsInPlace(op_plus, new MatrixBlock(coords_mul.rlen, coords_mul.clen, 1.0));

        ReorgOperator op_t = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), threads);
        // inx = t(coords[1,])
        MatrixBlock inx;
        inx = coords_mul.slice(0,0);
        inx = inx.reorgOperations(op_t, new MatrixBlock(), 0,0,inx.getNumColumns());
        // iny = t(coords[2,])
        MatrixBlock iny;
        iny = coords_mul.slice(1,1);
        iny = iny.reorgOperations(op_t, new MatrixBlock(), 0,0,iny.getNumColumns());
        //System.out.println(iny);
        // # any out-of-range pixels, if present, correspond to an extra pixel with fill_value at the end of the input
        // index_vector = (orig_w *(iny-1) + inx) * ((0<inx) & (inx<=orig_w) & (0<iny) & (iny<=orig_h))
        BinaryOperator op_minus = InstructionUtils.parseExtendedBinaryOperator("-");
        BinaryOperator op_mult = InstructionUtils.parseExtendedBinaryOperator("*");
        /*
         Nx1 matrix of the second term of the above equation for multiplying later on with the first part
         ((0<inx) & (inx<=orig_w) & (0<iny) & (iny<=orig_h))
         */
        //System.out.println(inx);
        BinaryOperator op_greater = InstructionUtils.parseExtendedBinaryOperator(">");
        //BinaryOperator op_less = InstructionUtils.parseExtendedBinaryOperator("<");
        BinaryOperator op_less_equal = InstructionUtils.parseExtendedBinaryOperator("<=");
        //BinaryOperator op_greater_equal = InstructionUtils.parseExtendedBinaryOperator(">=");
        BinaryOperator op_and = InstructionUtils.parseExtendedBinaryOperator("&&");
        MatrixBlock helper_one; //(0<inx)
        helper_one = inx.binaryOperations(op_greater, new MatrixBlock(1,1,0.0));
        MatrixBlock helper_two; //(inx<=orig_w)
        helper_two = inx.binaryOperations(op_less_equal, new MatrixBlock(1,1,(double) orig_w));
        MatrixBlock helper_three; //(0<iny)
        helper_three = iny.binaryOperations(op_greater, new MatrixBlock(1,1, 0.0));
        MatrixBlock helper_four;
        helper_four = iny.binaryOperations(op_less_equal, new MatrixBlock(1,1, (double) orig_h));
        MatrixBlock second_term;
        second_term = helper_one.binaryOperations(op_and, helper_two); //(0<inx) & (inx<=orig_w)
        second_term.binaryOperationsInPlace(op_and, helper_three); //(0<inx) & (inx<=orig_w) & (0<iny)
        second_term.binaryOperationsInPlace(op_and, helper_four); // (0<inx) & (inx<=orig_w) & (0<iny) & (iny<=orig_h)

        //System.out.println(second_term);


        // Nx1 matrix as the first part of the equation for the index vector (orig_w *(iny-1) + inx)
        MatrixBlock index_vector;
        index_vector = iny.binaryOperations(op_minus, new MatrixBlock(1, 1, 1.0)); //(iny-1)
        index_vector.binaryOperationsInPlace(op_mult, new MatrixBlock(1, 1, (double) orig_w)); //orig_w *(iny-1)
        index_vector.binaryOperationsInPlace(op_plus, inx); // (orig_w *(iny-1) + inx)
        index_vector.binaryOperationsInPlace(op_mult, second_term); //(orig_w *(iny-1) + inx) * ((0<inx) & (inx<=orig_w) & (0<iny) & (iny<=orig_h))
        index_vector = index_vector.reorgOperations(op_t, new MatrixBlock(), 0,0,index_vector.getNumRows());
        //System.out.println(inx);
        System.out.println(index_vector);

        // xs = ((index_vector == 0)*(orig_w*orig_h +1)) + index_vector
        BinaryOperator op_equal = InstructionUtils.parseExtendedBinaryOperator("==");
        helper_one = index_vector.binaryOperations(op_equal, new MatrixBlock(1,1, 0.0)); //(index_vector == 0)
        helper_one = helper_one.binaryOperations(op_mult, new MatrixBlock(1,1, (double) (orig_w*orig_h+1))); //((index_vector == 0)*(orig_w*orig_h +1))
        //System.out.println(helper_one);

        MatrixBlock xs;
        xs = helper_one.binaryOperations(op_plus, index_vector); //xs = ((index_vector == 0)*(orig_w*orig_h +1)) + index_vector
        System.out.println(xs);

        //#if(min(index_vector) == 0){
        //#  ys=cbind(img_in, matrix(fill_value,nrow(img_in), 1))
        //#}else{
        //#  ys = img_in
        //#}
        //get output for the condition above so that the fillvalue can be added outside the transformation Matrix method
        MatrixBlock fillBlock;
        if (index_vector.min() == 0){
            fillBlock = new MatrixBlock(1, 1, 1.0);
        }else{
            fillBlock = new MatrixBlock(1, 1, 0.0);
        }
        System.out.println(fillBlock);

        /**ind= matrix(seq(1,ncol(xs),1),1,ncol(xs))
        * z = table(xs, ind)
        * zMat = transMat
        */
        //
        return new MatrixBlock[] {index_vector, fillBlock};
    }
}
