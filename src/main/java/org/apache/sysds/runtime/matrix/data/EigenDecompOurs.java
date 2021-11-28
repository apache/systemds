package org.apache.sysds.runtime.matrix.data;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.operators.*;
import org.apache.sysds.runtime.util.DataConverter;

// TODO: EigenDecompostition entirely in SystemDS matrix-operators

// NOTES: what does org.apache.commons.math3.linear.EigenDecomposition use?
//			- only for real matrices (symmetric and non-symmetric)
//			- we only use the real valued eigenvalues
//			- This implementation is based on the paper "The Implicit QL Algorithm" (1971)
//			- similar to JAMA implementation
//
//	      Apache common source
//			- https://gitbox.apache.org/repos/asf?p=commons-math.git
//
//		  Symmetric:
//			- tred2 (p.217) -> tql2 (p.244)	(list of procedures p. 192)
//		  Non-symmetric:
//			- othes (p.349) (or dirhes, elmhes) -> hqr2 (p. 383)
//
//        Make sure to return the EVec and EVal sorted by values of Eval descending order
//
//        Look at power iterations
//        Use Matrixblock and parallelism
//        One QL and one for e.g. symmetric
//        scripts/staging/lanczos

public class EigenDecompOurs {
    private double[][] m;

    public EigenDecompOurs(MatrixBlock in) {
        if ( in.getNumRows() != in.getNumColumns() ) {
            throw new DMLRuntimeException("Eigen Decomposition can only be done on a square matrix. "
                    + "Input matrix is rectangular (rows=" + in.getNumRows() + ", cols="+ in.getNumColumns() +")");
        }
        this.m= DataConverter.convertToArray2DRowRealMatrix(in).getData();
        if(isSym(this.m)) {
            Lanczos(in);
        }

    }

    private boolean isSym(double[][] m) {

        for(int i = 0; i < m.length; i++) {
            for(int j = 0; j < m[i].length; j++) {
                if (Double.compare(m[i][j], m[j][i]) != 0)
                    return false;
            }
        }
        return true;
    }


    private void Lanczos(MatrixBlock A) {
        int num_Threads = 1;

        int m = A.getNumRows();
        MatrixBlock v0 = new MatrixBlock(m, 1, 0.0);
        MatrixBlock v1 = MatrixBlock.randOperations(m, 1, 1.0, 0.0, 1.0, "UNIFORM", 0xC0FFEE);

        // normalize v1
        double v1_sum = v1.sum();
        RightScalarOperator op_div_scalar = new RightScalarOperator(Divide.getDivideFnObject(), v1_sum, num_Threads);
        v1 = v1.scalarOperations(op_div_scalar, new MatrixBlock());
        UnaryOperator op_sqrt = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.SQRT), num_Threads, true);
        v1 = v1.unaryOperations(op_sqrt, new MatrixBlock());
        assert v1.sumSq() == 1.0 : "v1 not correctly normalized";

        MatrixBlock T = new MatrixBlock(m, m, 0.0);
        MatrixBlock TV = new MatrixBlock(m, 1, 0.0);
        MatrixBlock w1;

        BinaryOperator op_mul = new BinaryOperator(Multiply.getMultiplyFnObject(), num_Threads);
        ReorgOperator op_t = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), num_Threads);
        TernaryOperator op_minus_mul = new TernaryOperator(MinusMultiply.getFnObject(), num_Threads);
        AggregateBinaryOperator op_mul_agg = InstructionUtils.getMatMultOperator(num_Threads);

        MatrixBlock beta = new MatrixBlock(1, 1, 0.0);
        for(int i = 0; i < m; i++) {
            if (i == 0)
                TV.copy(v1);
            else
                TV = TV.append(v1, new MatrixBlock(), true);

            w1 = A.aggregateBinaryOperations(A, v1, op_mul_agg);
            MatrixBlock w1_t = w1.reorgOperations(op_t,new MatrixBlock(), 0, 0, m);
            MatrixBlock alpha = w1_t.aggregateBinaryOperations(w1_t, v1, op_mul_agg);
            if(i < m-1) {
                w1 = w1.ternaryOperations(op_minus_mul, v1, alpha, new MatrixBlock());
                w1 = w1.ternaryOperations(op_minus_mul, v0, beta, new MatrixBlock());
                beta.setValue(0, 0, Math.sqrt(w1.sumSq()));
                v0.copy(v1);
                op_div_scalar = (RightScalarOperator)op_div_scalar.setConstant(beta.getDouble(0, 0));
                v1 = w1.scalarOperations(op_div_scalar, new MatrixBlock());

                T.setValue(i+1, i, beta.getValue(0, 0));
                T.setValue(i, i+1, beta.getValue(0, 0));
            }
            T.setValue(i, i, alpha.getValue(0, 0));
        }

        // TEST
        MatrixBlock[] a = LibCommonsMath.multiReturnOperations(A, "eigen");
        MatrixBlock[] b = LibCommonsMath.multiReturnOperations(T, "eigen");

        MatrixBlock evec = TV.aggregateBinaryOperations(TV, b[1], op_mul_agg);
    }

    public MatrixBlock getV() {
        return null;
    }

    public MatrixBlock getRealEigenvalues() {
        return null;
    }
}

