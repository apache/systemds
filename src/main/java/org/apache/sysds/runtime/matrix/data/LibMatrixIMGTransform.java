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

package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.CTable;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;

import static org.apache.commons.math3.util.FastMath.floor;

/**Separate class for generating the transformation matrix based on an affine transformation matrix and a matrix
 * containing the dimensions of the input and the target dimensions of the output
 */
public class LibMatrixIMGTransform {

    protected static final Log LOG = LogFactory.getLog(LibMatrixIMGTransform.class.getName());

    /** This method produces a transformation matrix for affine transformation (see Wikipedia: "Affine transformation").
     * It takes a transformation 3x3 matrix (last row is always 0,0,1 for images), called affine matrix, and a dimension matrix, where the
     * original image dimensions (width x height) are stored alongside the output image dimensions (width x height)
     * @param transMat affine 3x3 matrix for image transformations
     * @param dimMat 2x2 matrix with original and output image dimensions
     * @param threads number of threads for use in different methods
     *
     * @return array of two matrix blocks, 1st is the transformation matrix, 2nd a 1x1 matrix with 1 or 0
     */
    public static MatrixBlock[] transformationMatrix(MatrixBlock transMat, MatrixBlock dimMat, int threads) {
        //check the correctness of the input dimension matrix
        isValidDimensionMatrix(dimMat);

        int orig_w = (int) dimMat.get(0,0);
        int orig_h = (int) dimMat.get(0,1);
        int out_w = (int) dimMat.get(1,0);
        int out_h = (int) dimMat.get(1,1);

        //calculate the inverse of the transformation matrix
        MatrixBlock t_Inv = LibCommonsMath.unaryOperations(transMat, "inverse");

        //create the coords matrix: coords = matrix(1, rows=3, cols=out_w*out_h)
        MatrixBlock coords = new MatrixBlock(3, out_w*out_h, false);
        //change values to coords[1, ] = t((seq(0, out_w * out_h - 1) %% out_w) + 0.5)
        //coords[2, ] = t((seq(0, out_w * out_h - 1) %/% out_w) + 0.5)
        double [] coords1 = new double[out_w*out_h];
        double [] coords2 = new double[out_w*out_h];
        double [] coords3 = new double[out_w*out_h];
        for(int i=0; i<out_w*out_h; i++) {
            coords1[i] = (i % out_w) + 0.5;
            coords2[i] = floor((double) i / out_w) + 0.5;
            coords3[i] = 1.0;
        }
        coords.init(new double[][] {coords1, coords2, coords3}, 3, out_w*out_h);

        //# compute sampling pixel indices
        // coords = floor(T_inv %*% coords) + 1; coords in this instance coords_mul
        MatrixBlock coords_mul;
        assert t_Inv != null;
        AggregateBinaryOperator op_mul_agg = InstructionUtils.getMatMultOperator(threads);
        UnaryOperator op_floor = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.FLOOR));
        BinaryOperator op_plus = InstructionUtils.parseExtendedBinaryOperator("+");
        //(T_inv %*% coords)
        coords_mul = t_Inv.aggregateBinaryOperations(t_Inv, coords, op_mul_agg);
        //floor(T_inv %*% coords)
        coords_mul = coords_mul.unaryOperations(op_floor);
        //coords = floor(T_inv %*% coords) + 1
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

        // # any out-of-range pixels, if present, correspond to an extra pixel with fill_value at the end of the input
        // index_vector = (orig_w *(iny-1) + inx) * ((0<inx) & (inx<=orig_w) & (0<iny) & (iny<=orig_h))
        BinaryOperator op_minus = InstructionUtils.parseExtendedBinaryOperator("-");
        BinaryOperator op_mult = InstructionUtils.parseExtendedBinaryOperator("*");
        /*
         Nx1 matrix of the second term of the above equation for multiplying later on with the first part
         ((0<inx) & (inx<=orig_w) & (0<iny) & (iny<=orig_h))
         */
        BinaryOperator op_greater = InstructionUtils.parseExtendedBinaryOperator(">");
        BinaryOperator op_less_equal = InstructionUtils.parseExtendedBinaryOperator("<=");
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
        //(0<inx) & (inx<=orig_w)
        second_term = helper_one.binaryOperations(op_and, helper_two);
        //(0<inx) & (inx<=orig_w) & (0<iny)
        second_term.binaryOperationsInPlace(op_and, helper_three);
        // (0<inx) & (inx<=orig_w) & (0<iny) & (iny<=orig_h)
        second_term.binaryOperationsInPlace(op_and, helper_four);

        // Nx1 matrix as the first part of the equation for the index vector (orig_w *(iny-1) + inx)
        MatrixBlock index_vector;
        //(iny-1)
        index_vector = iny.binaryOperations(op_minus, new MatrixBlock(1, 1, 1.0));
        //orig_w *(iny-1)
        index_vector.binaryOperationsInPlace(op_mult, new MatrixBlock(1, 1, (double) orig_w));
        // (orig_w *(iny-1) + inx)
        index_vector.binaryOperationsInPlace(op_plus, inx);
        //(orig_w *(iny-1) + inx) * ((0<inx) & (inx<=orig_w) & (0<iny) & (iny<=orig_h))
        index_vector.binaryOperationsInPlace(op_mult, second_term);
        index_vector = index_vector.reorgOperations(op_t, new MatrixBlock(), 0,0,index_vector.getNumRows());

        // xs = ((index_vector == 0)*(orig_w*orig_h +1)) + index_vector
        BinaryOperator op_equal = InstructionUtils.parseExtendedBinaryOperator("==");
        //(index_vector == 0)
        helper_one = index_vector.binaryOperations(op_equal, new MatrixBlock(1,1, 0.0));
        //((index_vector == 0)*(orig_w*orig_h +1))
        helper_one = helper_one.binaryOperations(op_mult, new MatrixBlock(1,1, (double) (orig_w*orig_h+1)));
        MatrixBlock xs;
        //xs = ((index_vector == 0)*(orig_w*orig_h +1)) + index_vector
        xs = helper_one.binaryOperations(op_plus, index_vector);

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

        // ind= matrix(seq(1,ncol(xs),1),1,ncol(xs))
        // z = table(xs, ind)
        // zMat = transMat
        // generate an index vector/matrix and use a contingency table on xs and the generated index vector
        //ind= matrix(seq(1,ncol(xs),1),1,ncol(xs))
        double [] inds = new double[xs.getNumColumns()];
        for(int i=0; i<xs.getNumColumns(); i++) {
            inds[i] = i+1;
        }
        //ind= matrix(seq(1,ncol(xs),1),1,ncol(xs))
        MatrixBlock ind = new MatrixBlock(1, xs.getNumColumns(), inds);

        // get a ctable object to be able to generate a contingeny table (ctable)
        // this does not seem to be possible in another way, e.g. by using operators like above
        CTable ctab = CTable.getCTableFnObject();
        // create a ctable map where the results will be stored
        CTableMap m1 = new CTableMap();
        // create a matrix to store the (later converted) ctable map results
        MatrixBlock zMat;
        // run the ctable.execute loop for each entry of both vectors
        // w is the weight assigned to each observation and set to 1 to not influence values
        for( int i=0; i<xs.getNumColumns(); i++){
                double v1 = xs.get(0,i);
                double v2 = ind.get(0, i);
                double w = 1.0;
                ctab.execute(v1,v2,w,false, m1);
        }
        // one of two ways to handle ctable maps in order to generate a matrix block
        zMat = DataConverter.convertToMatrixBlock(m1);
        //return the transformation matrix as well as a matrix block with size 1x1 with
        //a double thats either 1 or 0 depending on the following comparison
        // index_vector.min() == 0
        // if it is true (the double value is 1.0) then: ys = cbind(img_in, matrix(fill_value,nrow(img_in), 1))
        // if it is false (the double value is 0.0) then ys = img_in
        return new MatrixBlock[] {zMat, fillBlock};
    }

    /** Validates the values of the dimension matrix for the affine transformation algorithm
     * Values can only be positive natural numbers (i.e. 1,2,3..) as matrices are constructed
     * based on the dimensions
     * @param dimMat 2x2 matrix with original and output image dimensions
     */
    private static void isValidDimensionMatrix(MatrixBlock dimMat){
        //check if the values of the dimension matrix are equal or above one, otherwise it is not a valid image
        if(dimMat.get(0,0)>=1 && dimMat.get(0,1)>=1 && dimMat.get(1,0)>=1 && dimMat.get(1,1)>=1){
            //check if the double values of the dimension matrix are actually positive natural numbers
            //i.e. that they can be cast to int without loss of information for matrix generation
            if(!(dimMat.get(0,0)== floor(dimMat.get(0,0))) || !(dimMat.get(0,1) == floor(dimMat.get(0,1)))
            || !(dimMat.get(1,0)== floor(dimMat.get(1,0))) || !(dimMat.get(1,1) == floor(dimMat.get(1,1)))){
                throw new RuntimeException("Image dimensions are not positive natural numbers! Check input and output image dimensions!");
            }
        }else{
            throw new RuntimeException("Wrong values! Image dimensions cannot be zero or negative! Check input and output image dimensions!");
        }
    }
}
