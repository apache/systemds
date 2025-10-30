package org.apache.sysds.runtime.einsum;

import org.apache.commons.logging.Log;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;

import java.util.ArrayList;

import static org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction.ensureMatrixBlockColumnVector;
import static org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction.ensureMatrixBlockRowVector;

public class EOpNodeBinary extends EOpNode {

    public enum EBinaryOperand { // upper case: char has to remain, lower case: to be summed
        ////// summations:   //////
        aB_a,// -> B
        Ba_a, // -> B
        Ba_aC, // mmult -> BC
        aB_Ca,
        Ba_Ca, // -> BC
        aB_aC, // outer mult, possibly with transposing first -> BC
        a_a,// dot ->

        ////// elementwisemult and sums, something like ij,ij->i   //////
        aB_aB,// elemwise and colsum -> B
        Ba_Ba, // elemwise and rowsum ->B
        Ba_aB, // elemwise, either colsum or rowsum -> B
		aB_Ba,

        ////// elementwise, no summations:   //////
        A_A,// v-elemwise -> A
        AB_AB,// M-M elemwise -> AB
        AB_BA, // M-M.T elemwise -> AB
        AB_A, // M-v colwise -> BA!?
        BA_A, // M-v rowwise -> BA
        ab_ab,//M-M sum all
        ab_ba, //M-M.T sum all
        ////// other   //////
        A_B, // outer mult -> AB
        A_scalar, // v-scalar
        AB_scalar, // m-scalar
        scalar_scalar
    }
    public EOpNode _left;
    public EOpNode _right;
    public EBinaryOperand _operand;
    public EOpNodeBinary(Character c1, Character c2, EOpNode left, EOpNode right, EBinaryOperand operand){
        super(c1,c2);
        this._left = left;
        this._right = right;
        this._operand = operand;
    }

    @Override
    public MatrixBlock computeEOpNode(ArrayList<MatrixBlock> inputs, int numThreads, Log LOG) {
        EOpNodeBinary bin = this;
        MatrixBlock left = _left.computeEOpNode(inputs, numThreads, LOG);
        MatrixBlock right = _right.computeEOpNode(inputs, numThreads, LOG);

        AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());

        MatrixBlock res;

        if(LOG.isTraceEnabled()) LOG.trace("computing binary "+bin._left +","+bin._right +"->"+bin);

        switch (bin._operand){
            case AB_AB -> {
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
            }
            case A_A -> {
                ensureMatrixBlockColumnVector(left);
                ensureMatrixBlockColumnVector(right);
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
            }
            case a_a -> {
                ensureMatrixBlockColumnVector(left);
                ensureMatrixBlockColumnVector(right);
				res = new MatrixBlock(0.0);
				res.allocateDenseBlock();
				res.getDenseBlockValues()[0] = LibMatrixMult.dotProduct(left.getDenseBlockValues(), right.getDenseBlockValues(), 0,0 , left.getNumRows());
            }
            case Ba_Ba -> {
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
                AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
                res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
            }
            case aB_aB -> {
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
                AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
                res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
                ensureMatrixBlockColumnVector(res);
            }
            case ab_ab -> {
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
                AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
                res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
            }
            case ab_ba -> {
                ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
                right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
                AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
                res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
            }
            case Ba_aB -> {
                ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
                right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
                AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
                res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
            }
            case aB_Ba -> {
                ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
                left = left.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
                AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
                res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
            }

            case AB_BA -> {
                ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
                right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
            }
            case Ba_aC -> {
                res = LibMatrixMult.matrixMult(left,right, new MatrixBlock(), numThreads);
            }
            case aB_Ca -> {
                res = LibMatrixMult.matrixMult(right,left, new MatrixBlock(), numThreads);
            }
            case Ba_Ca -> {
                ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
                right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
                res = LibMatrixMult.matrixMult(left,right, new MatrixBlock(), numThreads);
            }
            case aB_aC -> {
                if(false && LibMatrixMult.isSkinnyRightHandSide(left.getNumRows(), left.getNumColumns(), right.getNumRows(), right.getNumColumns(), true)){
                    res = new MatrixBlock(left.getNumColumns(), right.getNumColumns(),false);
                    res.allocateDenseBlock();
                    double[] m1 = left.getDenseBlock().values(0);
                    double[] m2 = right.getDenseBlock().values(0);
                    double[] c = res.getDenseBlock().values(0);
                    int alen = left.getNumColumns();
                    int blen = right.getNumColumns();
                    for(int i =0;i<left.getNumRows();i++){
                        LibSpoofPrimitives.vectOuterMultAdd(m1,m2,c,i*alen,i*blen, 0,alen,blen);
                    }
                }else {
                    ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
                    left = left.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
                    res = LibMatrixMult.matrixMult(left, right, new MatrixBlock(), numThreads);
                }
            }
            case A_scalar, AB_scalar -> {
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left},new ScalarObject[]{new DoubleObject(right.get(0,0))}, new MatrixBlock());
            }
            case BA_A -> {
                ensureMatrixBlockRowVector(right);
                res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
            }
            case Ba_a -> {
                ensureMatrixBlockRowVector(right);
                res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
                AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
                res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
            }

            case AB_A -> {
                ensureMatrixBlockColumnVector(right);
                res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
            }
            case aB_a -> {
                ensureMatrixBlockColumnVector(right);
                res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
                AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
                res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
                ensureMatrixBlockColumnVector(res);
            }

            case A_B -> {
                ensureMatrixBlockColumnVector(left);
                ensureMatrixBlockRowVector(right);
                res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
            }
            case scalar_scalar -> {
                return new MatrixBlock(left.get(0,0)*right.get(0,0));
            }
            default -> {
                throw new IllegalArgumentException("Unexpected value: " + bin._operand.toString());
            }

        }
        return res;
    }

    @Override
    public void reorderChildren(Character outChar1, Character outChar2) {
        if (this._operand==EBinaryOperand.aB_aC){
            if(this._right.c2 == outChar1) {
                    var tmp = _left;
                    _left = _right;
                    _right = tmp;
                    var tmp2 = c1;
                    c1 = c2;
                    c2 = tmp2;
            }
            _left.reorderChildren(_left.c2, _left.c1);
            // check if change happened:
            if(_left.c2 == _right.c1) {
                this._operand = EBinaryOperand.Ba_aC;
            }
        }
    }

}
