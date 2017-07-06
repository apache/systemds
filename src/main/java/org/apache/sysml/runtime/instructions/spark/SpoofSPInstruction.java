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

package org.apache.sysml.runtime.instructions.spark;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.codegen.CodegenUtils;
import org.apache.sysml.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysml.runtime.codegen.SpoofCellwise;
import org.apache.sysml.runtime.codegen.SpoofMultiAggregate;
import org.apache.sysml.runtime.codegen.SpoofCellwise.AggOp;
import org.apache.sysml.runtime.codegen.SpoofCellwise.CellType;
import org.apache.sysml.runtime.codegen.SpoofOperator;
import org.apache.sysml.runtime.codegen.SpoofOuterProduct;
import org.apache.sysml.runtime.codegen.SpoofOuterProduct.OutProdType;
import org.apache.sysml.runtime.codegen.SpoofRowwise;
import org.apache.sysml.runtime.codegen.SpoofRowwise.RowType;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.spark.SPInstruction;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;

import scala.Tuple2;

public class SpoofSPInstruction extends SPInstruction
{
	private final Class<?> _class;
	private final byte[] _classBytes;
	private final CPOperand[] _in;
	private final CPOperand _out;
	
	public SpoofSPInstruction(Class<?> cls, byte[] classBytes, CPOperand[] in, CPOperand out, String opcode, String str) {
		super(opcode, str);
		_class = cls;
		_classBytes = classBytes;
		_sptype = SPINSTRUCTION_TYPE.SpoofFused;
		_in = in;
		_out = out;
	}
	
	public static SpoofSPInstruction parseInstruction(String str) 
		throws DMLRuntimeException
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		
		//String opcode = parts[0];
		ArrayList<CPOperand> inlist = new ArrayList<CPOperand>();
		Class<?> cls = CodegenUtils.getClass(parts[1]);
		byte[] classBytes = CodegenUtils.getClassData(parts[1]);
		String opcode =  parts[0] + CodegenUtils.createInstance(cls).getSpoofType();
		
		for( int i=2; i<parts.length-2; i++ )
			inlist.add(new CPOperand(parts[i]));
		CPOperand out = new CPOperand(parts[parts.length-2]);
		//note: number of threads parts[parts.length-1] always ignored
		
		return new SpoofSPInstruction(cls, classBytes, inlist.toArray(new CPOperand[0]), out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException 
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;

		//get input rdd and variable name
		ArrayList<String> bcVars = new ArrayList<String>();
		MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(_in[0].getName());
		JavaPairRDD<MatrixIndexes, MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( _in[0].getName() );
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = null;
				
		//simple case: map-side only operation (one rdd input, broadcast all)
		//keep track of broadcast variables
		ArrayList<PartitionedBroadcast<MatrixBlock>> bcMatrices = new ArrayList<PartitionedBroadcast<MatrixBlock>>();
		ArrayList<ScalarObject> scalars = new ArrayList<ScalarObject>();
		for( int i=1; i<_in.length; i++ ) {
			if( _in[i].getDataType()==DataType.MATRIX) {
				bcMatrices.add(sec.getBroadcastForVariable(_in[i].getName()));
				bcVars.add(_in[i].getName());
			}
			else if(_in[i].getDataType()==DataType.SCALAR) {
				//note: even if literal, it might be compiled as scalar placeholder
				scalars.add(sec.getScalarInput(_in[i].getName(), _in[i].getValueType(), _in[i].isLiteral()));
			}
		}
		
		//initialize Spark Operator
		if(_class.getSuperclass() == SpoofCellwise.class) // cellwise operator
		{
			SpoofCellwise op = (SpoofCellwise) CodegenUtils.createInstance(_class); 	
			AggregateOperator aggop = getAggregateOperator(op.getAggOp());
			
			if( _out.getDataType()==DataType.MATRIX ) {
				out = in.mapPartitionsToPair(new CellwiseFunction(_class.getName(), _classBytes, bcMatrices, scalars), true);
				if( op.getCellType()==CellType.ROW_AGG && mcIn.getCols() > mcIn.getColsPerBlock() ) {
					//TODO investigate if some other side effect of correct blocks
					if( out.partitions().size() > mcIn.getNumRowBlocks() )
						out = RDDAggregateUtils.aggByKeyStable(out, aggop, (int)mcIn.getNumRowBlocks(), false);
					else
						out = RDDAggregateUtils.aggByKeyStable(out, aggop, false);
				}
				sec.setRDDHandleForVariable(_out.getName(), out);
				
				//maintain lineage information for output rdd
				sec.addLineageRDD(_out.getName(), _in[0].getName());
				for( String bcVar : bcVars )
					sec.addLineageBroadcast(_out.getName(), bcVar);
				
				//update matrix characteristics
				updateOutputMatrixCharacteristics(sec, op);	
			}
			else { //SCALAR
				out = in.mapPartitionsToPair(new CellwiseFunction(_class.getName(), _classBytes, bcMatrices, scalars), true);
				MatrixBlock tmpMB = RDDAggregateUtils.aggStable(out, aggop);
				sec.setVariable(_out.getName(), new DoubleObject(tmpMB.getValue(0, 0)));
			}
		}
		else if(_class.getSuperclass() == SpoofMultiAggregate.class)
		{
			SpoofMultiAggregate op = (SpoofMultiAggregate) CodegenUtils.createInstance(_class); 	
			AggOp[] aggOps = op.getAggOps();
			
			MatrixBlock tmpMB = in
					.mapToPair(new MultiAggregateFunction(_class.getName(), _classBytes, bcMatrices, scalars))
					.values().fold(new MatrixBlock(), new MultiAggAggregateFunction(aggOps) );
			
			sec.setMatrixOutput(_out.getName(), tmpMB);
			return;
		}
		else if(_class.getSuperclass() == SpoofOuterProduct.class) // outer product operator
		{
			if( _out.getDataType()==DataType.MATRIX ) {
				SpoofOperator op = (SpoofOperator) CodegenUtils.createInstance(_class); 	
				OutProdType type = ((SpoofOuterProduct)op).getOuterProdType();

				//update matrix characteristics
				updateOutputMatrixCharacteristics(sec, op);			
				MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(_out.getName());
				
				out = in.mapPartitionsToPair(new OuterProductFunction(_class.getName(), _classBytes, bcMatrices, scalars), true);
				if(type == OutProdType.LEFT_OUTER_PRODUCT || type == OutProdType.RIGHT_OUTER_PRODUCT ) {
					//TODO investigate if some other side effect of correct blocks
					if( in.partitions().size() > mcOut.getNumRowBlocks()*mcOut.getNumColBlocks() )
						out = RDDAggregateUtils.sumByKeyStable(out, (int)(mcOut.getNumRowBlocks()*mcOut.getNumColBlocks()), false);
					else
						out = RDDAggregateUtils.sumByKeyStable(out, false);	
				}
				sec.setRDDHandleForVariable(_out.getName(), out);
				
				//maintain lineage information for output rdd
				sec.addLineageRDD(_out.getName(), _in[0].getName());
				for( String bcVar : bcVars )
					sec.addLineageBroadcast(_out.getName(), bcVar);
				
			}
			else {
				out = in.mapPartitionsToPair(new OuterProductFunction(_class.getName(), _classBytes, bcMatrices, scalars), true);
				MatrixBlock tmp = RDDAggregateUtils.sumStable(out);
				sec.setVariable(_out.getName(), new DoubleObject(tmp.getValue(0, 0)));
			}
		}
		else if( _class.getSuperclass() == SpoofRowwise.class ) { //row aggregate operator
			SpoofRowwise op = (SpoofRowwise) CodegenUtils.createInstance(_class); 	
			RowwiseFunction fmmc = new RowwiseFunction(_class.getName(), _classBytes, bcMatrices, scalars, (int)mcIn.getCols());
			out = in.mapPartitionsToPair(fmmc, op.getRowType()==RowType.ROW_AGG
					|| op.getRowType() == RowType.NO_AGG);
			
			if( op.getRowType().isColumnAgg() || op.getRowType()==RowType.FULL_AGG ) {
				MatrixBlock tmpMB = RDDAggregateUtils.sumStable(out);
				if( op.getRowType().isColumnAgg() )
					sec.setMatrixOutput(_out.getName(), tmpMB);
				else
					sec.setScalarOutput(_out.getName(), 
						new DoubleObject(tmpMB.quickGetValue(0, 0)));
			}
			else //row-agg or no-agg 
			{
				if( op.getRowType()==RowType.ROW_AGG && mcIn.getCols() > mcIn.getColsPerBlock() ) {
					//TODO investigate if some other side effect of correct blocks
					if( out.partitions().size() > mcIn.getNumRowBlocks() )
						out = RDDAggregateUtils.sumByKeyStable(out, (int)mcIn.getNumRowBlocks(), false);
					else
						out = RDDAggregateUtils.sumByKeyStable(out, false);
				}
				
				sec.setRDDHandleForVariable(_out.getName(), out);
				
				//maintain lineage information for output rdd
				sec.addLineageRDD(_out.getName(), _in[0].getName());
				for( String bcVar : bcVars )
					sec.addLineageBroadcast(_out.getName(), bcVar);
				
				//update matrix characteristics
				updateOutputMatrixCharacteristics(sec, op);
			}
			return;
		}
		else {
			throw new DMLRuntimeException("Operator " + _class.getSuperclass() + " is not supported on Spark");
		}
	}
	
	private void updateOutputMatrixCharacteristics(SparkExecutionContext sec, SpoofOperator op) 
		throws DMLRuntimeException 
	{
		if(op instanceof SpoofCellwise)
		{
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(_in[0].getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(_out.getName());
			if( ((SpoofCellwise)op).getCellType()==CellType.ROW_AGG )
				mcOut.set(mcIn.getRows(), 1, mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
			else if( ((SpoofCellwise)op).getCellType()==CellType.NO_AGG )
				mcOut.set(mcIn);
		}
		else if(op instanceof SpoofOuterProduct)
		{
			MatrixCharacteristics mcIn1 = sec.getMatrixCharacteristics(_in[0].getName()); //X
			MatrixCharacteristics mcIn2 = sec.getMatrixCharacteristics(_in[1].getName()); //U
			MatrixCharacteristics mcIn3 = sec.getMatrixCharacteristics(_in[2].getName()); //V
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(_out.getName());
			OutProdType type = ((SpoofOuterProduct)op).getOuterProdType();
			
			if( type == OutProdType.CELLWISE_OUTER_PRODUCT)
				mcOut.set(mcIn1.getRows(), mcIn1.getCols(), mcIn1.getRowsPerBlock(), mcIn1.getColsPerBlock());
			else if( type == OutProdType.LEFT_OUTER_PRODUCT) 		
				mcOut.set(mcIn3.getRows(), mcIn3.getCols(), mcIn3.getRowsPerBlock(), mcIn3.getColsPerBlock());		
			else if( type == OutProdType.RIGHT_OUTER_PRODUCT )
				mcOut.set(mcIn2.getRows(), mcIn2.getCols(), mcIn2.getRowsPerBlock(), mcIn2.getColsPerBlock());
		}
		else if(op instanceof SpoofRowwise) {
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(_in[0].getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(_out.getName());
			RowType type = ((SpoofRowwise)op).getRowType();
			if( type == RowType.NO_AGG )
				mcOut.set(mcIn);
			else if( type == RowType.ROW_AGG )
				mcOut.set(mcIn.getRows(), ((SpoofRowwise)op).isCBind0()? 2:1, 
					mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
			else if( type == RowType.COL_AGG )
				mcOut.set(1, mcIn.getCols(), mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
			else if( type == RowType.COL_AGG_T )
				mcOut.set(mcIn.getCols(), 1, mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
		}
	}
		
	private static class RowwiseFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -7926980450209760212L;

		private final ArrayList<PartitionedBroadcast<MatrixBlock>> _vectors;
		private final ArrayList<ScalarObject> _scalars;
		private final byte[] _classBytes;
		private final String _className;
		private final int _clen;
		private SpoofRowwise _op = null;
		
		public RowwiseFunction(String className, byte[] classBytes, ArrayList<PartitionedBroadcast<MatrixBlock>> bcMatrices, ArrayList<ScalarObject> scalars, int clen) 
			throws DMLRuntimeException
		{			
			_className = className;
			_classBytes = classBytes;
			_vectors = bcMatrices;
			_scalars = scalars;
			_clen = clen;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg ) 
			throws Exception 
		{
			//lazy load of shipped class
			if( _op == null ) {
				Class<?> loadedClass = CodegenUtils.getClass(_className, _classBytes);
				_op = (SpoofRowwise) CodegenUtils.createInstance(loadedClass); 
			}
			
			//setup local memory for reuse
			int clen2 = (int) (_op.getRowType().isRowTypeB1() ? _vectors.get(0).getNumCols() : -1);
			LibSpoofPrimitives.setupThreadLocalMemory(_op.getNumIntermediates(), _clen, clen2);
			
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			boolean aggIncr = (_op.getRowType().isColumnAgg() //aggregate entire partition
				|| _op.getRowType() == RowType.FULL_AGG); 
			MatrixBlock blkOut = aggIncr ? new MatrixBlock() : null;
			
			while( arg.hasNext() ) {
				//get main input block and indexes
				Tuple2<MatrixIndexes,MatrixBlock> e = arg.next();
				MatrixIndexes ixIn = e._1();
				MatrixBlock blkIn = e._2();
				int rowIx = (int)ixIn.getRowIndex();
				
				//prepare output and execute single-threaded operator
				ArrayList<MatrixBlock> inputs = getVectorInputsFromBroadcast(blkIn, rowIx);
				blkOut = aggIncr ? blkOut : new MatrixBlock();
				_op.execute(inputs, _scalars, blkOut, false, aggIncr);
				if( !aggIncr ) {
					MatrixIndexes ixOut = new MatrixIndexes(ixIn.getRowIndex(),
						_op.getRowType()!=RowType.NO_AGG ? 1 : ixIn.getColumnIndex());
					ret.add(new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut));
				}
			}
			
			//cleanup and final result preparations
			LibSpoofPrimitives.cleanupThreadLocalMemory();
			if( aggIncr )
				ret.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(1,1), blkOut));
			
			return ret.iterator();
		}
		
		private ArrayList<MatrixBlock> getVectorInputsFromBroadcast(MatrixBlock blkIn, int rowIndex) 
			throws DMLRuntimeException 
		{
			ArrayList<MatrixBlock> ret = new ArrayList<MatrixBlock>();
			ret.add(blkIn);
			for( PartitionedBroadcast<MatrixBlock> vector : _vectors )
				ret.add(vector.getBlock((vector.getNumRowBlocks()>=rowIndex)?rowIndex:1, 1));
			return ret;
		}
	}
	
	private static class CellwiseFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -8209188316939435099L;
		
		private ArrayList<PartitionedBroadcast<MatrixBlock>> _vectors = null;
		private ArrayList<ScalarObject> _scalars = null;
		private byte[] _classBytes = null;
		private String _className = null;
		private SpoofOperator _op = null;
		
		public CellwiseFunction(String className, byte[] classBytes, ArrayList<PartitionedBroadcast<MatrixBlock>> bcMatrices, ArrayList<ScalarObject> scalars) 
			throws DMLRuntimeException
		{
			_className = className;
			_classBytes = classBytes;
			_vectors = bcMatrices;
			_scalars = scalars;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg)
			throws Exception 
		{
			//lazy load of shipped class
			if( _op == null ) {
				Class<?> loadedClass = CodegenUtils.getClass(_className, _classBytes);
				_op = (SpoofOperator) CodegenUtils.createInstance(loadedClass); 
			}
			
			List<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			while(arg.hasNext()) 
			{
				Tuple2<MatrixIndexes,MatrixBlock> tmp = arg.next();
				MatrixIndexes ixIn = tmp._1();
				MatrixBlock blkIn = tmp._2();
				MatrixIndexes ixOut = ixIn; 
				MatrixBlock blkOut = new MatrixBlock();
				ArrayList<MatrixBlock> inputs = getVectorInputsFromBroadcast(blkIn, ixIn);
					
				//execute core operation
				if(((SpoofCellwise)_op).getCellType()==CellType.FULL_AGG) {
					ScalarObject obj = _op.execute(inputs, _scalars, 1);
					blkOut.reset(1, 1);
					blkOut.quickSetValue(0, 0, obj.getDoubleValue());	
				}
				else {
					if(((SpoofCellwise)_op).getCellType()==CellType.ROW_AGG)
						ixOut = new MatrixIndexes(ixOut.getRowIndex(), 1);
					_op.execute(inputs, _scalars, blkOut);
				}
				ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ixOut, blkOut));
			}
			return ret.iterator();
		}
		
		private ArrayList<MatrixBlock> getVectorInputsFromBroadcast(MatrixBlock blkIn, MatrixIndexes ixIn) 
			throws DMLRuntimeException 
		{
			ArrayList<MatrixBlock> ret = new ArrayList<MatrixBlock>();
			ret.add(blkIn);
			for( PartitionedBroadcast<MatrixBlock> in : _vectors ) {
				int rowIndex = (int)((in.getNumRowBlocks()>=ixIn.getRowIndex())?ixIn.getRowIndex():1);
				int colIndex = (int)((in.getNumColumnBlocks()>=ixIn.getColumnIndex())?ixIn.getColumnIndex():1);
				ret.add(in.getBlock(rowIndex, colIndex));
			}
			return ret;
		}
	}
	
	private static class MultiAggregateFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -5224519291577332734L;
		
		private ArrayList<PartitionedBroadcast<MatrixBlock>> _vectors = null;
		private ArrayList<ScalarObject> _scalars = null;
		private byte[] _classBytes = null;
		private String _className = null;
		private SpoofOperator _op = null;
		
		public MultiAggregateFunction(String className, byte[] classBytes, ArrayList<PartitionedBroadcast<MatrixBlock>> bcMatrices, ArrayList<ScalarObject> scalars) 
			throws DMLRuntimeException
		{
			_className = className;
			_classBytes = classBytes;
			_vectors = bcMatrices;
			_scalars = scalars;
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg)
			throws Exception 
		{
			//lazy load of shipped class
			if( _op == null ) {
				Class<?> loadedClass = CodegenUtils.getClass(_className, _classBytes);
				_op = (SpoofOperator) CodegenUtils.createInstance(loadedClass); 
			}
				
			//execute core operation
			ArrayList<MatrixBlock> inputs = getVectorInputsFromBroadcast(arg._2(), arg._1());
			MatrixBlock blkOut = new MatrixBlock();
			_op.execute(inputs, _scalars, blkOut);
			
			return new Tuple2<MatrixIndexes,MatrixBlock>(arg._1(), blkOut);
		}
		
		private ArrayList<MatrixBlock> getVectorInputsFromBroadcast(MatrixBlock blkIn, MatrixIndexes ixIn) 
			throws DMLRuntimeException 
		{
			ArrayList<MatrixBlock> ret = new ArrayList<MatrixBlock>();
			ret.add(blkIn);
			for( PartitionedBroadcast<MatrixBlock> in : _vectors ) {
				int rowIndex = (int)((in.getNumRowBlocks()>=ixIn.getRowIndex())?ixIn.getRowIndex():1);
				int colIndex = (int)((in.getNumColumnBlocks()>=ixIn.getColumnIndex())?ixIn.getColumnIndex():1);
				ret.add(in.getBlock(rowIndex, colIndex));
			}
			return ret;
		}
	}
	
	private static class MultiAggAggregateFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 5978731867787952513L;
		
		private AggOp[] _ops = null;
		
		public MultiAggAggregateFunction( AggOp[] ops ) {
			_ops = ops;	
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
			throws Exception 
		{
			//prepare combiner block
			if( arg0.getNumRows() <= 0 || arg0.getNumColumns() <= 0) {
				arg0.copy(arg1);
				return arg0;
			}
			else if( arg1.getNumRows() <= 0 || arg1.getNumColumns() <= 0 ) {
				return arg0;
			}
			
			//aggregate second input (in-place)
			SpoofMultiAggregate.aggregatePartialResults(_ops, arg0, arg1);
			
			return arg0;
		}
	}
	
	private static class OuterProductFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -8209188316939435099L;
		
		private ArrayList<PartitionedBroadcast<MatrixBlock>> _bcMatrices = null;
		private ArrayList<ScalarObject> _scalars = null;
		private byte[] _classBytes = null;
		private String _className = null;
		private SpoofOperator _op = null;
		
		public OuterProductFunction(String className, byte[] classBytes, ArrayList<PartitionedBroadcast<MatrixBlock>> bcMatrices, ArrayList<ScalarObject> scalars) 
				throws DMLRuntimeException
		{
			_className = className;
			_classBytes = classBytes;
			_bcMatrices = bcMatrices;
			_scalars = scalars;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg)
			throws Exception 
		{
			//lazy load of shipped class
			if( _op == null ) {
				Class<?> loadedClass = CodegenUtils.getClass(_className, _classBytes);
				_op = (SpoofOperator) CodegenUtils.createInstance(loadedClass); 
			}
			
			List<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			while(arg.hasNext())
			{
				Tuple2<MatrixIndexes,MatrixBlock> tmp = arg.next();
				MatrixIndexes ixIn = tmp._1();
				MatrixBlock blkIn = tmp._2();
				MatrixBlock blkOut = new MatrixBlock();

				ArrayList<MatrixBlock> inputs = new ArrayList<MatrixBlock>();
				inputs.add(blkIn);
				inputs.add(_bcMatrices.get(0).getBlock((int)ixIn.getRowIndex(), 1)); // U
				inputs.add(_bcMatrices.get(1).getBlock((int)ixIn.getColumnIndex(), 1)); // V
						
				//execute core operation
				if(((SpoofOuterProduct)_op).getOuterProdType()==OutProdType.AGG_OUTER_PRODUCT) {
					ScalarObject obj = _op.execute(inputs, _scalars,1);
					blkOut.reset(1, 1);
					blkOut.quickSetValue(0, 0, obj.getDoubleValue());
				}
				else {
					_op.execute(inputs, _scalars, blkOut);
				}
				
				ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(createOutputIndexes(ixIn,_op), blkOut));				
			}
			
			return ret.iterator();
		}
		
		private MatrixIndexes createOutputIndexes(MatrixIndexes in, SpoofOperator spoofOp) {
			if( ((SpoofOuterProduct)spoofOp).getOuterProdType() == OutProdType.LEFT_OUTER_PRODUCT ) 
				return new MatrixIndexes(in.getColumnIndex(), 1);
			else if ( ((SpoofOuterProduct)spoofOp).getOuterProdType() == OutProdType.RIGHT_OUTER_PRODUCT)
				return new MatrixIndexes(in.getRowIndex(), 1);
			else 
				return in;
		}		
	}
	
	public static AggregateOperator getAggregateOperator(AggOp aggop) {
		if( aggop == AggOp.SUM || aggop == AggOp.SUM_SQ )
			return new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.NONE);
		else if( aggop == AggOp.MIN )
			return new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject(BuiltinCode.MIN), false, CorrectionLocationType.NONE);
		else if( aggop == AggOp.MAX )
			return new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject(BuiltinCode.MAX), false, CorrectionLocationType.NONE);
		return null;
	}
}
