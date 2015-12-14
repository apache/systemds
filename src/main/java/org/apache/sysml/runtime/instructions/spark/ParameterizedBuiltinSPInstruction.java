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
import java.util.HashMap;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.ParameterizedBuiltinFunctionExpression;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.ParameterizedBuiltin;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.mr.GroupedAggregateInstruction;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcastMatrix;
import org.apache.sysml.runtime.instructions.spark.functions.ExtractGroup;
import org.apache.sysml.runtime.instructions.spark.functions.ExtractGroupNWeights;
import org.apache.sysml.runtime.instructions.spark.functions.PerformGroupByAggInCombiner;
import org.apache.sysml.runtime.instructions.spark.functions.PerformGroupByAggInReducer;
import org.apache.sysml.runtime.instructions.spark.functions.ReplicateVectorFunction;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.WeightedCell;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.CMOperator;
import org.apache.sysml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;
import org.apache.sysml.runtime.transform.DataTransform;
import org.apache.sysml.runtime.util.UtilFunctions;

public class ParameterizedBuiltinSPInstruction  extends ComputationSPInstruction 
{
	
	private int arity;
	protected HashMap<String,String> params;
	private boolean _bRmEmptyBC = false;
	
	public ParameterizedBuiltinSPInstruction(Operator op, HashMap<String,String> paramsMap, CPOperand out, String opcode, String istr, boolean bRmEmptyBC )
	{
		super(op, null, null, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.ParameterizedBuiltin;
		params = paramsMap;
		_bRmEmptyBC = bRmEmptyBC;
	}

	public int getArity() {
		return arity;
	}
	
	public HashMap<String,String> getParams() { return params; }
	
	public static HashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		HashMap<String,String> paramMap = new HashMap<String,String>();
		
		// all parameters are of form <name=value>
		String[] parts;
		for ( int i=1; i <= params.length-2; i++ ) {
			parts = params[i].split(Lop.NAME_VALUE_SEPARATOR);
			paramMap.put(parts[0], parts[1]);
		}
		
		return paramMap;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand( parts[parts.length-1] ); 

		// process remaining parts and build a hash map
		HashMap<String,String> paramsMap = constructParameterMap(parts);

		// determine the appropriate value function
		ValueFunction func = null;
		if ( opcode.equalsIgnoreCase("groupedagg")) {
			// check for mandatory arguments
			String fnStr = paramsMap.get("fn");
			if ( fnStr == null ) 
				throw new DMLRuntimeException("Function parameter is missing in groupedAggregate.");
			if ( fnStr.equalsIgnoreCase("centralmoment") ) {
				if ( paramsMap.get("order") == null )
					throw new DMLRuntimeException("Mandatory \"order\" must be specified when fn=\"centralmoment\" in groupedAggregate.");
			}
			
			Operator op = GroupedAggregateInstruction.parseGroupedAggOperator(fnStr, paramsMap.get("order"));
			return new ParameterizedBuiltinSPInstruction(op, paramsMap, out, opcode, str, false);
		}
		else if(   opcode.equalsIgnoreCase("rmempty") ) 
		{
			boolean bRmEmptyBC = false; 
			if(parts.length > 6)
				bRmEmptyBC = (parts[5].compareTo("true") == 0)?true:false;
								
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinSPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str, bRmEmptyBC);
		}
		else if(   opcode.equalsIgnoreCase("rexpand") ) 
		{
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinSPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str, false);
		}
		else if(   opcode.equalsIgnoreCase("replace") ) 
		{
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinSPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str, false);
		}
		else if ( opcode.equalsIgnoreCase("transform") ) 
		{
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			String specFile = paramsMap.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_TXSPEC);
			String applyTxPath = paramsMap.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_APPLYMTD);
			if ( specFile != null && applyTxPath != null)
				throw new DMLRuntimeException(
						"Invalid parameters to transform(). Only one of '"
								+ ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_TXSPEC
								+ "' or '"
								+ ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_APPLYMTD
								+ "' can be specified.");
			return new ParameterizedBuiltinSPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str, false);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode (" + opcode + ") for ParameterizedBuiltin Instruction.");
		}

	}
	

	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		String opcode = getOpcode();
		
		//opcode guaranteed to be a valid opcode (see parsing)
		if ( opcode.equalsIgnoreCase("groupedagg") ) 
		{	
			//get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> target = sec.getBinaryBlockRDDHandleForVariable( params.get(Statement.GAGG_TARGET) );
			JavaPairRDD<MatrixIndexes,MatrixBlock> groups = sec.getBinaryBlockRDDHandleForVariable( params.get(Statement.GAGG_GROUPS) );
			JavaPairRDD<MatrixIndexes,MatrixBlock> weights = null;
			
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics( params.get(Statement.GAGG_TARGET) );
			MatrixCharacteristics mc2 = sec.getMatrixCharacteristics( params.get(Statement.GAGG_GROUPS) );
			if(mc1.dimsKnown() && mc2.dimsKnown() && (mc1.getRows() != mc2.getRows() || mc1.getCols() != mc2.getCols())) {
				throw new DMLRuntimeException("Grouped Aggregate SPInstruction is not supported for dimension of target != groups");
			}
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			
			JavaPairRDD<Long, WeightedCell> groupWeightedCells = null;
			
			// Step 1: First extract groupWeightedCells from group, target and weights
			if ( params.get(Statement.GAGG_WEIGHTS) != null ) {
				weights = sec.getBinaryBlockRDDHandleForVariable( params.get(Statement.GAGG_WEIGHTS) );
				
				MatrixCharacteristics mc3 = sec.getMatrixCharacteristics( params.get(Statement.GAGG_GROUPS) );
				if(mc1.dimsKnown() && mc3.dimsKnown() && (mc1.getRows() != mc3.getRows() || mc1.getCols() != mc3.getCols())) {
					throw new DMLRuntimeException("Grouped Aggregate SPInstruction is not supported for dimension of target != weights");
				}
				
				groupWeightedCells = groups.join(target).join(weights)
						.flatMapToPair(new ExtractGroupNWeights());	
			}
			else {
				groupWeightedCells = groups.join(target)
						.flatMapToPair(new ExtractGroup());
			}
			
			// Step 2: Make sure we have brlen required while creating <MatrixIndexes, MatrixCell> 
			if(mc1.getRowsPerBlock() == -1) {
				throw new DMLRuntimeException("The block sizes are not specified for grouped aggregate");
			}
			int brlen = mc1.getRowsPerBlock();
			
			// Step 3: Now perform grouped aggregate operation (either on combiner side or reducer side)
			JavaPairRDD<MatrixIndexes, MatrixCell> out = null;
			if(_optr instanceof CMOperator && ((CMOperator) _optr).isPartialAggregateOperator() ) {
				out = groupWeightedCells.reduceByKey(new PerformGroupByAggInCombiner(_optr))
						.mapToPair(new CreateMatrixCell(brlen, _optr));
			}
			else {
				// Use groupby key because partial aggregation is not supported
				out = groupWeightedCells.groupByKey()
						.mapToPair(new PerformGroupByAggInReducer(_optr))
						.mapToPair(new CreateMatrixCell(brlen, _optr));
			}
			
			// Step 4: Set output characteristics and rdd handle 
			setOutputCharacteristicsForGroupedAgg(mc1, mcOut, out);
			
			//store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			
			sec.addLineageRDD(output.getName(), params.get(Statement.GAGG_TARGET) );
			sec.addLineageRDD(output.getName(), params.get(Statement.GAGG_GROUPS) );
			if ( params.get(Statement.GAGG_WEIGHTS) != null ) {
				sec.addLineageRDD(output.getName(), params.get(Statement.GAGG_WEIGHTS) );
			}
		}
		else if ( opcode.equalsIgnoreCase("rmempty") ) 
		{
			String rddInVar = params.get("target");
			String rddOffVar = params.get("offset");
			
			//get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( rddInVar );
			JavaPairRDD<MatrixIndexes,MatrixBlock> off;
			PartitionedBroadcastMatrix broadcastOff;
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(rddInVar);
			boolean rows = sec.getScalarInput(params.get("margin"), ValueType.STRING, true).getStringValue().equals("rows");
			long maxDim = sec.getScalarInput(params.get("maxdim"), ValueType.DOUBLE, false).getLongValue();
			long brlen = mcIn.getRowsPerBlock();
			long bclen = mcIn.getColsPerBlock();
			long numRep = (long)Math.ceil( rows ? (double)mcIn.getCols()/bclen : (double)mcIn.getRows()/brlen);
			
			//execute remove empty rows/cols operation
			JavaPairRDD<MatrixIndexes,MatrixBlock> out;

			if(_bRmEmptyBC){
				broadcastOff = sec.getBroadcastForVariable(rddOffVar );
				// Broadcast offset vector
				out = in
					.flatMapToPair(new RDDRemoveEmptyFunctionInMem(rows, maxDim, brlen, bclen, broadcastOff));		
			}
			else {
				off = sec.getBinaryBlockRDDHandleForVariable( rddOffVar );
				out = in
					.join( off.flatMapToPair(new ReplicateVectorFunction(!rows,numRep)) )
					.flatMapToPair(new RDDRemoveEmptyFunction(rows, maxDim, brlen, bclen));		
			}				

			out = RDDAggregateUtils.mergeByKey(out);
			
			//store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddInVar);
			if(!_bRmEmptyBC)
				sec.addLineageRDD(output.getName(), rddOffVar);
			else
				sec.addLineageBroadcast(output.getName(), rddOffVar);		// TODO
			
			//update output statistics (required for correctness)
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			mcOut.set(rows?maxDim:mcIn.getRows(), rows?mcIn.getCols():maxDim, (int)brlen, (int)bclen, mcIn.getNonZeros());
		}
		else if ( opcode.equalsIgnoreCase("replace") ) 
		{	
			//get input rdd handle
			String rddVar = params.get("target");
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar );
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(rddVar);
			
			//execute replace operation
			double pattern = Double.parseDouble( params.get("pattern") );
			double replacement = Double.parseDouble( params.get("replacement") );
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = 
					in1.mapValues(new RDDReplaceFunction(pattern, replacement));
			
			//store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddVar);
			
			//update output statistics (required for correctness)
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			mcOut.set(mcIn.getRows(), mcIn.getCols(), mcIn.getRowsPerBlock(), mcIn.getColsPerBlock(), (pattern!=0 && replacement!=0)?mcIn.getNonZeros():-1);
		}
		else if ( opcode.equalsIgnoreCase("rexpand") ) 
		{
			String rddInVar = params.get("target");
			
			//get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( rddInVar );
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(rddInVar);
			double maxVal = Double.parseDouble( params.get("max") );
			long lmaxVal = UtilFunctions.toLong(maxVal);
			boolean dirRows = params.get("dir").equals("rows");
			boolean cast = Boolean.parseBoolean(params.get("cast"));
			boolean ignore = Boolean.parseBoolean(params.get("ignore"));
			long brlen = mcIn.getRowsPerBlock();
			long bclen = mcIn.getColsPerBlock();
			
			//execute remove empty rows/cols operation
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = in
					.flatMapToPair(new RDDRExpandFunction(maxVal, dirRows, cast, ignore, brlen, bclen));		
			out = RDDAggregateUtils.mergeByKey(out);
			
			//store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddInVar);
			
			//update output statistics (required for correctness)
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			mcOut.set(dirRows?lmaxVal:mcIn.getRows(), dirRows?mcIn.getRows():lmaxVal, (int)brlen, (int)bclen, -1);
		}
		else if ( opcode.equalsIgnoreCase("transform") ) 
		{
			// perform data transform on Spark
			try {
				DataTransform.spDataTransform(
						this, 
						new MatrixObject[] { (MatrixObject) sec.getVariable(params.get("target")) }, 
						new MatrixObject[] { (MatrixObject) sec.getVariable(output.getName()) }, ec);
			} catch (Exception e) {
				throw new DMLRuntimeException(e);
			}
		}
	}
	

	/**
	 * 
	 */
	public static class RDDReplaceFunction implements Function<MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 6576713401901671659L;
		
		private double _pattern; 
		private double _replacement;
		
		public RDDReplaceFunction(double pattern, double replacement) 
		{
			_pattern = pattern;
			_replacement = replacement;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0) 
			throws Exception 
		{
			return (MatrixBlock) arg0.replaceOperations(new MatrixBlock(), _pattern, _replacement);
		}		
	}
	
	/**
	 * 
	 */
	public static class RDDRemoveEmptyFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,Tuple2<MatrixBlock, MatrixBlock>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 4906304771183325289L;

		private boolean _rmRows; 
		private long _len;
		private long _brlen;
		private long _bclen;
				
		public RDDRemoveEmptyFunction(boolean rmRows, long len, long brlen, long bclen) 
		{
			_rmRows = rmRows;
			_len = len;
			_brlen = brlen;
			_bclen = bclen;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> arg0)
			throws Exception 
		{
			//prepare inputs (for internal api compatibility)
			IndexedMatrixValue data = SparkUtils.toIndexedMatrixBlock(arg0._1(),arg0._2()._1());
			IndexedMatrixValue offsets = SparkUtils.toIndexedMatrixBlock(arg0._1(),arg0._2()._2());
			
			//execute remove empty operations
			ArrayList<IndexedMatrixValue> out = new ArrayList<IndexedMatrixValue>();
			LibMatrixReorg.rmempty(data, offsets, _rmRows, _len, _brlen, _bclen, out);

			//prepare and return outputs
			return SparkUtils.fromIndexedMatrixBlock(out);
		}
	}
	
	/**
	 * 
	 */
	public static class RDDRemoveEmptyFunctionInMem implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 4906304771183325289L;

		private boolean _rmRows; 
		private long _len;
		private long _brlen;
		private long _bclen;
		
		private PartitionedBroadcastMatrix _off = null;
				
		public RDDRemoveEmptyFunctionInMem(boolean rmRows, long len, long brlen, long bclen, PartitionedBroadcastMatrix off) 
		{
			_rmRows = rmRows;
			_len = len;
			_brlen = brlen;
			_bclen = bclen;
			_off = off;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception 
		{
			//prepare inputs (for internal api compatibility)
			IndexedMatrixValue data = SparkUtils.toIndexedMatrixBlock(arg0._1(),arg0._2());
			//IndexedMatrixValue offsets = SparkUtils.toIndexedMatrixBlock(arg0._1(),arg0._2()._2());
			IndexedMatrixValue offsets = null;
			if(_rmRows)
				offsets = SparkUtils.toIndexedMatrixBlock(arg0._1(), _off.getMatrixBlock((int)arg0._1().getRowIndex(), 1));
			else
				offsets = SparkUtils.toIndexedMatrixBlock(arg0._1(), _off.getMatrixBlock(1, (int)arg0._1().getColumnIndex()));
			
			//execute remove empty operations
			ArrayList<IndexedMatrixValue> out = new ArrayList<IndexedMatrixValue>();
			LibMatrixReorg.rmempty(data, offsets, _rmRows, _len, _brlen, _bclen, out);

			//prepare and return outputs
			return SparkUtils.fromIndexedMatrixBlock(out);
		}
	}
	
	/**
	 * 
	 */
	public static class RDDRExpandFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = -6153643261956222601L;
		
		private double _maxVal;
		private boolean _dirRows;
		private boolean _cast;
		private boolean _ignore;
		private long _brlen;
		private long _bclen;
		
		public RDDRExpandFunction(double maxVal, boolean dirRows, boolean cast, boolean ignore, long brlen, long bclen) 
		{
			_maxVal = maxVal;
			_dirRows = dirRows;
			_cast = cast;
			_ignore = ignore;
			_brlen = brlen;
			_bclen = bclen;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception 
		{
			//prepare inputs (for internal api compatibility)
			IndexedMatrixValue data = SparkUtils.toIndexedMatrixBlock(arg0._1(),arg0._2());
			
			//execute rexpand operations
			ArrayList<IndexedMatrixValue> out = new ArrayList<IndexedMatrixValue>();
			LibMatrixReorg.rexpand(data, _maxVal, _dirRows, _cast, _ignore, _brlen, _bclen, out);
			
			//prepare and return outputs
			return SparkUtils.fromIndexedMatrixBlock(out);
		}
	}
	
	/**
	 * 
	 */
	public static class CreateMatrixCell implements PairFunction<Tuple2<Long,WeightedCell>, MatrixIndexes, MatrixCell> 
	{
		private static final long serialVersionUID = -5783727852453040737L;
		
		int brlen; Operator op;
		public CreateMatrixCell(int brlen, Operator op) {
			this.brlen = brlen;
			this.op = op;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixCell> call(Tuple2<Long, WeightedCell> kv) throws Exception {

			double val = -1;
			if(op instanceof CMOperator)
			{
				AggregateOperationTypes agg=((CMOperator)op).aggOpType;
				switch(agg)
				{
				case COUNT:
					val = kv._2.getWeight();
					break;
				case MEAN:
					val = kv._2.getValue();
					break;
				case CM2:
					val = kv._2.getValue()/ kv._2.getWeight();
					break;
				case CM3:
					val = kv._2.getValue()/ kv._2.getWeight();
					break;
				case CM4:
					val = kv._2.getValue()/ kv._2.getWeight();
					break;
				case VARIANCE:
					val = kv._2.getValue()/kv._2.getWeight();
					// val = kv._2.getWeight() ==1.0? 0:kv._2.getValue()/(kv._2.getWeight() - 1);
					break;
				default:
					throw new DMLRuntimeException("Invalid aggreagte in CM_CV_Object: " + agg);
				}
			}
			else
			{
				//avoid division by 0
				val = kv._2.getValue()/kv._2.getWeight();
			}
			
			MatrixIndexes indx = new MatrixIndexes(kv._1, 1);
			MatrixCell cell = new MatrixCell(val);
			
			return new Tuple2<MatrixIndexes, MatrixCell>(indx, cell);
		}
		
	}
	
	/**
	 * 
	 * @param mc1
	 * @param mcOut
	 * @param out
	 * @throws DMLRuntimeException
	 */
	public void setOutputCharacteristicsForGroupedAgg(MatrixCharacteristics mc1, MatrixCharacteristics mcOut, JavaPairRDD<MatrixIndexes, MatrixCell> out) 
		throws DMLRuntimeException 
	{
		if(!mcOut.dimsKnown()) {
			if(!mc1.dimsKnown()) {
				throw new DMLRuntimeException("The output dimensions are not specified for grouped aggregate");
			}
			
			if ( params.get(Statement.GAGG_NUM_GROUPS) != null) {
				int ngroups = (int) Double.parseDouble(params.get(Statement.GAGG_NUM_GROUPS));
				mcOut.set(ngroups, 1, mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			}
			else {
				out = SparkUtils.cacheBinaryCellRDD(out);
				mcOut.set(SparkUtils.computeMatrixCharacteristics(out));
				mcOut.setBlockSize(mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			}
		}
	}
	
}
