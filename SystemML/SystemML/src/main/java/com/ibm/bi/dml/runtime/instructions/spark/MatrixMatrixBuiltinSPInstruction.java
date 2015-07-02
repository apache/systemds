/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class MatrixMatrixBuiltinSPInstruction extends BuiltinBinarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixMatrixBuiltinSPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String opcode,
											   String istr){
		super(op, in1, in2, out, 2, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		
		SparkExecutionContext sec = (SparkExecutionContext) ec;
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown() && !mc1.dimsKnown() && !mc2.dimsKnown()) {
			throw new DMLRuntimeException("The output dimensions are not specified for MatrixMatrixBuiltinSPInstruction");
		}
		else if(mc1.getRows() != mc2.getRows() || mc1.getCols() != mc2.getCols()) {
			throw new DMLRuntimeException("Incorrect dimensions specified for MatrixMatrixBuiltinSPInstruction");
		}
		else if(!mcOut.dimsKnown()) {
			mcOut.set(mc1);
			// mcOut.setDimension(mc1.getRows(), mc1.getCols());
		}
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
		BinaryOperator bop = (BinaryOperator) _optr;
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.cogroup(in2).mapToPair(new StreamableMMBuiltin(bop));
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
		
	}
	
	public static class StreamableMMBuiltin implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<MatrixBlock>,Iterable<MatrixBlock>>>, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = -3761426075578633158L;
		
		BinaryOperator bop;
		public StreamableMMBuiltin(BinaryOperator bop) {
			this.bop = bop;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>> kv) throws Exception {
			MatrixBlock matBlock1 = null;
			MatrixBlock matBlock2 = null;
			for(MatrixBlock m : kv._2._1) {
				if(matBlock1 == null) {
					matBlock1 = m;
				}
				else {
					throw new Exception("ERROR: Multiple blocks for the given MatrixIndexes");
				}
			}
			for(MatrixBlock m : kv._2._2) {
				if(matBlock2 == null) {
					matBlock2 = m;
				}
				else {
					throw new Exception("ERROR: Multiple blocks for the given MatrixIndexes");
				}
			}
			
			MatrixBlock resultBlock = (MatrixBlock) matBlock1.binaryOperations(bop, matBlock2, new MatrixBlock());
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlock);
		}
		
	}
	
}