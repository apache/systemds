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

package org.apache.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import scala.Serializable;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class UnaryMatrixSPInstruction extends UnarySPInstruction {

	protected UnaryMatrixSPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		super(SPType.Unary, op, in, out, opcode, instr);
	}
	
	public static UnarySPInstruction parseInstruction ( String str ) {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseUnaryInstruction(str, in, out);
		return new UnaryMatrixSPInstruction(
			InstructionUtils.parseUnaryOperator(opcode), in, out, opcode, str);
	}

	@Override 
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		
		//execute unary builtin operation
		UnaryOperator uop = (UnaryOperator) _optr;
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in.mapValues(new RDDMatrixBuiltinUnaryOp(uop));
		
		//set output RDD
		updateUnaryOutputDataCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);	
		sec.addLineageRDD(output.getName(), input1.getName());

		//FIXME: implement similar to cumsum through 
		//  CumulativeAggregateSPInstruction (Spark)
		//  UnaryMatrixCPInstruction (local cumsum on aggregates)
		//  CumulativeOffsetSPInstruction (Spark)  
		if ( "urowcumk+".equals(getOpcode()) ) {

			JavaPairRDD< MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock> > localRowcumsum = in.mapToPair( new LocalRowCumsumFunction() );

			// Collect end-values of every block of every row for offset calc by grouping by global row index
			JavaPairRDD< Long, Iterable<Tuple3<Long, Long, double[]>> > rowEndValues = localRowcumsum
				.mapToPair( tuple2 -> {
					// get index of block
					MatrixIndexes indexes = tuple2._1;
					// get cum matrix block
					MatrixBlock localRowcumsumBlock = tuple2._2._2;

					// get row and column block index
					long rowBlockIndex = indexes.getRowIndex();
					long colBlockIndex = indexes.getColumnIndex();

					// Save end value of every row of every block (if block is empty save 0)
					double[] endValues = new double[ localRowcumsumBlock.getNumRows() ];

					for ( int i = 0; i < localRowcumsumBlock.getNumRows(); i ++ ) {
						if (localRowcumsumBlock.getNumColumns() > 0)
							endValues[i] = localRowcumsumBlock.get(i, localRowcumsumBlock.getNumColumns() - 1);
						else
							endValues[i] = 0.0 ;
					}
					return new Tuple2<>(rowBlockIndex, new Tuple3<>(rowBlockIndex, colBlockIndex, endValues));
				}
				).groupByKey();

			// compute offset for every block
			List< Tuple2 <Tuple2<Long, Long>, double[]> > offsetList = rowEndValues
				.flatMapToPair(tuple2 -> {
					Long rowBlockIndex = tuple2._1;
					List< Tuple3<Long, Long, double[]> > colValues = new ArrayList<>();
					for ( Tuple3<Long, Long, double[]> cv : tuple2._2 ) 
						colValues.add(cv);
					
					// sort blocks from one row by column index
					colValues.sort(Comparator.comparing(Tuple3::_2));

					// get number of rows of a block by counting amount of end (row) values of said block
					int numberOfRows = 0;
					if ( !colValues.isEmpty() ) {
						Tuple3<Long, Long, double[]> firstTuple = colValues.get(0);
						double[] lastValuesArray = firstTuple._3();
						numberOfRows = lastValuesArray.length;
					}

					List<Tuple2<Tuple2<Long, Long>, double[]>> blockOffsets = new ArrayList<>();
					double[] cumulativeOffsets = new double[numberOfRows];
					for (Tuple3<Long, Long, double[]> colValue : colValues) {
						Long colBlockIndex = colValue._2();
						double[] endValues = colValue._3();

						// copy current offsets
						double[] currentOffsets = cumulativeOffsets.clone();

						// and save block indexes with its offsets
						blockOffsets.add( new Tuple2<>(new Tuple2<>(rowBlockIndex, colBlockIndex), currentOffsets) );

						for ( int i = 0; i < numberOfRows && i < endValues.length; i++ ) {
							cumulativeOffsets[i] += endValues[i];
						}
					}
					return blockOffsets.iterator();
				}
				).collect();

			// convert list to map for easier access to offsets
			Map< Tuple2<Long, Long>, double[] > offsetMap = new HashMap<>();
			for (Tuple2<Tuple2<Long, Long>, double[]> offset : offsetList) {
				offsetMap.put(offset._1, offset._2);
			}

			out = localRowcumsum.mapToPair( new FinalRowCumsumFunction(offsetMap)) ;

			updateUnaryOutputDataCharacteristics(sec);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
	}



	private static class LocalRowCumsumFunction implements PairFunction< Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock> > {
		private static final long serialVersionUID = 2388003441846068046L;

		@Override
		public Tuple2< MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock> > call(Tuple2<MatrixIndexes, MatrixBlock> tuple2) {


			MatrixBlock inputBlock = tuple2._2;
			MatrixBlock cumsumBlock = new MatrixBlock( inputBlock.getNumRows(), inputBlock.getNumColumns(), false );


			for ( int i = 0; i < inputBlock.getNumRows(); i++ ) {

				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();

				for ( int j = 0; j < inputBlock.getNumColumns(); j++ ) {

					double val = inputBlock.get(i, j);
					kplus.execute2(kbuff, val);
					cumsumBlock.set(i, j, kbuff._sum);
				}
			}
			// original index, original matrix and local cumsum block
			return new Tuple2<>( tuple2._1, new Tuple2<>(inputBlock, cumsumBlock) );
		}
	}




	private static class FinalRowCumsumFunction implements PairFunction<Tuple2< MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock> >, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = -6738155890298916270L;
		// map block indexes to the row offsets
		private final Map< Tuple2<Long, Long>, double[] > offsetMap;

		public FinalRowCumsumFunction(Map<Tuple2<Long, Long>, double[]> offsetMap) {
			this.offsetMap = offsetMap;
		}


		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2< MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock> > tuple ) {

			MatrixIndexes indexes = tuple._1;
			MatrixBlock inputBlock = tuple._2._1;
			MatrixBlock localRowCumsumBlock = tuple._2._2;

			// key to get the row offset for this block
			Tuple2<Long, Long> blockKey = new Tuple2<>( indexes.getRowIndex(), indexes.getColumnIndex()) ;
			double[] offsets = offsetMap.get(blockKey);

			MatrixBlock cumsumBlock = new MatrixBlock( inputBlock.getNumRows(), inputBlock.getNumColumns(), false );


			for ( int i = 0; i < inputBlock.getNumRows(); i++ ) {

				double rowOffset = 0.0;
				if ( offsets != null && i < offsets.length ) {
					rowOffset = offsets[i];
				}

				for ( int j = 0; j < inputBlock.getNumColumns(); j++ ) {
					double cumsumValue = localRowCumsumBlock.get(i, j);
					cumsumBlock.set(i, j, cumsumValue + rowOffset);
				}
			}

			// block index and final cumsum block
			return new Tuple2<>(indexes, cumsumBlock);
		}
	}



	// helper class
	private static class Tuple3<Type1, Type2, Type3> implements Serializable {

		private static final long serialVersionUID = 123;
		private final Type2 _2;
		private final Type3 _3;


		public Tuple3( Type1 _1, Type2 _2, Type3 _3 ) {
			this._2 = _2;
			this._3 = _3;
		}

		public Type2 _2() {
			return _2;
		}

		public Type3 _3() {
			return _3;
		}
	}

	private static class RDDMatrixBuiltinUnaryOp implements Function<MatrixBlock,MatrixBlock> 
	{
		private static final long serialVersionUID = -3128192099832877491L;
		
		private UnaryOperator _op = null;
		
		public RDDMatrixBuiltinUnaryOp(UnaryOperator u_op) {
			_op = u_op;
		}

		@Override
		public MatrixBlock call(MatrixBlock arg0) throws Exception {
			return arg0.unaryOperations(_op, new MatrixBlock());
		}
	}
}

