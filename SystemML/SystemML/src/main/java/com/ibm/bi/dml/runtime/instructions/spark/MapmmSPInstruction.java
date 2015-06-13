/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;


import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.api.DMLException; 
import com.ibm.bi.dml.hops.AggBinaryOp.SparkAggType;
import com.ibm.bi.dml.lops.MapMult;
import com.ibm.bi.dml.lops.MapMult.CacheType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.AggregateSumMultiBlockFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.AggregateSumSingleBlockFunction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * TODO: pre-filter and post-filter of empty blocks (if applicable) via rdd.filter()
 * TODO: generalized mapmult for multiple output blocks per input block
 * TODO: destroy (cleanup) of broadcast variables; we cannot clean them up right away, because no computation
 *       has been triggered yet 
 * TODO: we need to reason about multiple broadcast variables for chains of mapmults (sum of operations until cleanup) 
 * 
 */
public class MapmmSPInstruction extends BinarySPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private CacheType _type = null;
	//private boolean _outputEmpty = true;
	private SparkAggType _aggtype;
	
	public MapmmSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, CacheType type, 
			                    boolean outputEmpty, SparkAggType aggtype, String opcode, String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.MAPMM;
		
		_type = type;
		//_outputEmpty = outputEmpty;
		_aggtype = aggtype;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MapmmSPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String opcode = InstructionUtils.getOpCode(str);

		if ( opcode.equalsIgnoreCase(MapMult.OPCODE)) {
			String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
			in1.split(parts[1]);
			in2.split(parts[2]);
			out.split(parts[3]);
			CacheType type = CacheType.valueOf(parts[4]);
			boolean outputEmpty = Boolean.parseBoolean(parts[5]);
			SparkAggType aggtype = SparkAggType.valueOf(parts[6]);
			
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			return new MapmmSPInstruction(aggbin, in1, in2, out, type, outputEmpty, aggtype, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("MapmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		String rddVar = (_type==CacheType.LEFT) ? input2.getName() : input1.getName();
		String bcastVar = (_type==CacheType.LEFT) ? input1.getName() : input2.getName();
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(output.getName());
		
		//get inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar );
		Broadcast<MatrixBlock> in2 = sec.getBroadcastForVariable( bcastVar ); 
		
		//execute mapmult instruction
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.mapToPair( new RDDMapMMFunction(_type, in2, mc.getRowsPerBlock(), mc.getColsPerBlock()) );
		
		//perform aggregation if necessary and put output into symbol table
		if( _aggtype == SparkAggType.SINGLE_BLOCK )
		{
			MatrixBlock out2 = out.values()
					              .reduce(new AggregateSumSingleBlockFunction());
			
			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), out2);
		}
		else //MULTI_BLOCK or NONE
		{
			if( _aggtype == SparkAggType.MULTI_BLOCK )
				out = out.reduceByKey( new AggregateSumMultiBlockFunction() );
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddVar);
			sec.addLineageBroadcast(output.getName(), bcastVar);
			
			//update output statistics if not inferred
			updateBinaryMMOutputMatrixCharacteristics(sec);
		}
	}
	
	/**
	 * 
	 * 
	 */
	private static class RDDMapMMFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;

		private CacheType _type = null;
		private int _brlen = -1;
		private int _bclen = -1;
		
		private AggregateBinaryOperator _op = null;
		private MatrixBlock[] _partBlocks = null;
		
		public RDDMapMMFunction( CacheType type, Broadcast<MatrixBlock> binput, int brlen, int bclen )
		{
			_type = type;
			_brlen = brlen;
			_bclen = bclen;
			
			//get the broadcast vector
			MatrixBlock mb = binput.value();
			
			//partition vector for fast in memory lookup
			try
			{
				if( _type == CacheType.LEFT )
				{
					//in-memory colblock partitioning (according to brlen of rdd)
					int lclen = mb.getNumColumns();
					int numBlocks = (int)Math.ceil((double)lclen/_brlen);				
					_partBlocks = new MatrixBlock[numBlocks];
					for( int i=0; i<numBlocks; i++ )
					{
						MatrixBlock tmp = new MatrixBlock();
						mb.sliceOperations(1, mb.getNumRows(), 
								i*_brlen+1, Math.min((i+1)*_brlen, lclen),  tmp);
						_partBlocks[i] = tmp;
					}
				}
				else //if( _type == CacheType.RIGHT )
				{
					//in-memory rowblock partitioning (according to bclen of rdd)
					int lrlen = mb.getNumRows();
					int numBlocks = (int)Math.ceil((double)lrlen/_bclen);				
					_partBlocks = new MatrixBlock[numBlocks];
					for( int i=0; i<numBlocks; i++ )
					{
						MatrixBlock tmp = new MatrixBlock();
						mb.sliceOperations(i*_bclen+1, Math.min((i+1)*_bclen, lrlen), 
								1, mb.getNumColumns(), tmp);
						_partBlocks[i] = tmp;
					}						
				}
			}
			catch(DMLException ex)
			{
				LOG.error("Failed partitioning of broadcast variable input.", ex);
			}
			
			//created operator for reuse
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();
			
			if( _type == CacheType.LEFT )
			{
				//get the right hand side matrix
				MatrixBlock left = _partBlocks[(int)ixIn.getRowIndex()-1];
				
				//execute matrix-vector mult
				OperationsOnMatrixValues.performAggregateBinary( 
						new MatrixIndexes(1,ixIn.getRowIndex()), left, ixIn, blkIn, ixOut, blkOut, _op);						
			}
			else //if( _type == CacheType.RIGHT )
			{
				//get the right hand side matrix
				MatrixBlock right = _partBlocks[(int)ixIn.getColumnIndex()-1];
				
				//execute matrix-vector mult
				OperationsOnMatrixValues.performAggregateBinary(
						ixIn, blkIn, new MatrixIndexes(ixIn.getColumnIndex(),1), right, ixOut, blkOut, _op);					
			}
			
			
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
	}
	
}
