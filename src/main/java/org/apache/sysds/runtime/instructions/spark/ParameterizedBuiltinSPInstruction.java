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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.ParameterizedBuiltin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.instructions.spark.data.LazyIterableIterator;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.instructions.spark.functions.ExtractGroup.ExtractGroupBroadcast;
import org.apache.sysds.runtime.instructions.spark.functions.ExtractGroup.ExtractGroupJoin;
import org.apache.sysds.runtime.instructions.spark.functions.ExtractGroupNWeights;
import org.apache.sysds.runtime.instructions.spark.functions.PerformGroupByAggInCombiner;
import org.apache.sysds.runtime.instructions.spark.functions.PerformGroupByAggInReducer;
import org.apache.sysds.runtime.instructions.spark.functions.ReplicateVectorFunction;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixCell;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.data.WeightedCell;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBagOfWords;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.transform.meta.TfOffsetMap;
import org.apache.sysds.runtime.transform.tokenize.Tokenizer;
import org.apache.sysds.runtime.transform.tokenize.TokenizerFactory;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;

import org.apache.sysds.utils.stats.SparkStatistics;
import scala.Tuple2;

public class ParameterizedBuiltinSPInstruction extends ComputationSPInstruction {
	protected HashMap<String, String> params;

	ParameterizedBuiltinSPInstruction(Operator op, HashMap<String, String> paramsMap, CPOperand out, String opcode,
		String istr) {
		super(SPType.ParameterizedBuiltin, op, null, null, out, opcode, istr);
		params = paramsMap;
	}

	public HashMap<String, String> getParams() {
		return params;
	}

	public CacheableData<?> getTarget(ExecutionContext ec) {
		return ec.getCacheableData(params.get("target"));
	}

	public static HashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		HashMap<String, String> paramMap = new HashMap<>();

		// all parameters are of form <name=value>
		String[] parts;
		for(int i = 1; i <= params.length - 2; i++) {
			parts = params[i].split(Lop.NAME_VALUE_SEPARATOR);
			paramMap.put(parts[0], parts[1]);
		}

		return paramMap;
	}

	public static ParameterizedBuiltinSPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase(Opcodes.MAPGROUPEDAGG.toString())) {
			CPOperand target = new CPOperand(parts[1]);
			CPOperand groups = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);

			HashMap<String, String> paramsMap = new HashMap<>();
			paramsMap.put(Statement.GAGG_TARGET, target.getName());
			paramsMap.put(Statement.GAGG_GROUPS, groups.getName());
			paramsMap.put(Statement.GAGG_NUM_GROUPS, parts[4]);

			Operator op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.LASTCOLUMN);
			return new ParameterizedBuiltinSPInstruction(op, paramsMap, out, opcode, str);
		}
		else {
			// last part is always the output
			CPOperand out = new CPOperand(parts[parts.length - 1]);

			// process remaining parts and build a hash map
			HashMap<String, String> paramsMap = constructParameterMap(parts);

			// determine the appropriate value function
			ValueFunction func = null;
			if(opcode.equalsIgnoreCase(Opcodes.GROUPEDAGG.toString())) {
				// check for mandatory arguments
				String fnStr = paramsMap.get("fn");
				if(fnStr == null)
					throw new DMLRuntimeException("Function parameter is missing in groupedAggregate.");
				if(fnStr.equalsIgnoreCase("centralmoment")) {
					if(paramsMap.get("order") == null)
						throw new DMLRuntimeException(
							"Mandatory \"order\" must be specified when fn=\"centralmoment\" in groupedAggregate.");
				}
				Operator op = InstructionUtils.parseGroupedAggOperator(fnStr, paramsMap.get("order"));
				return new ParameterizedBuiltinSPInstruction(op, paramsMap, out, opcode, str);
			}
			else if(opcode.equalsIgnoreCase(Opcodes.RMEMPTY.toString())) {
				func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
				return new ParameterizedBuiltinSPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
			}
			else if(opcode.equalsIgnoreCase(Opcodes.REXPAND.toString()) || opcode.equalsIgnoreCase(Opcodes.REPLACE.toString()) ||
				opcode.equalsIgnoreCase(Opcodes.LOWERTRI.toString()) || opcode.equalsIgnoreCase(Opcodes.UPPERTRI.toString()) ||
				opcode.equalsIgnoreCase(Opcodes.TOKENIZE.toString()) || opcode.equalsIgnoreCase(Opcodes.TRANSFORMAPPLY.toString()) ||
				opcode.equalsIgnoreCase(Opcodes.TRANSFORMDECODE.toString())) {
				func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
				return new ParameterizedBuiltinSPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
			}
			else if(opcode.equalsIgnoreCase(Opcodes.CONTAINS.toString())) {
				return new ParameterizedBuiltinSPInstruction(null, paramsMap, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Unknown opcode (" + opcode + ") for ParameterizedBuiltin Instruction.");
			}
		}
	}

	@Override
	@SuppressWarnings("unchecked")
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext) ec;
		String opcode = getOpcode();

		// opcode guaranteed to be a valid opcode (see parsing)
		if(opcode.equalsIgnoreCase(Opcodes.MAPGROUPEDAGG.toString())) {
			// get input rdd handle
			String targetVar = params.get(Statement.GAGG_TARGET);
			String groupsVar = params.get(Statement.GAGG_GROUPS);
			JavaPairRDD<MatrixIndexes, MatrixBlock> target = sec.getBinaryMatrixBlockRDDHandleForVariable(targetVar);
			PartitionedBroadcast<MatrixBlock> groups = sec.getBroadcastForVariable(groupsVar);
			DataCharacteristics mc1 = sec.getDataCharacteristics(targetVar);
			DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
			CPOperand ngrpOp = new CPOperand(params.get(Statement.GAGG_NUM_GROUPS));
			int ngroups = (int) sec.getScalarInput(ngrpOp).getLongValue();

			// single-block aggregation
			if(ngroups <= mc1.getBlocksize() && mc1.getCols() <= mc1.getBlocksize()) {
				// execute map grouped aggregate
				JavaRDD<MatrixBlock> out = target.map(new RDDMapGroupedAggFunction2(groups, _optr, ngroups));
				MatrixBlock out2 = RDDAggregateUtils.sumStable(out);

				// put output block into symbol table (no lineage because single block)
				// this also includes implicit maintenance of matrix characteristics
				sec.setMatrixOutput(output.getName(), out2);
			}
			// multi-block aggregation
			else {
				// execute map grouped aggregate
				JavaPairRDD<MatrixIndexes, MatrixBlock> out = target
					.flatMapToPair(new RDDMapGroupedAggFunction(groups, _optr, ngroups, mc1.getBlocksize()));

				out = RDDAggregateUtils.sumByKeyStable(out, false);

				// updated characteristics and handle outputs
				mcOut.set(ngroups, mc1.getCols(), mc1.getBlocksize(), -1);
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), targetVar);
				sec.addLineageBroadcast(output.getName(), groupsVar);
			}
		}
		else if(opcode.equalsIgnoreCase(Opcodes.GROUPEDAGG.toString())) {
			boolean broadcastGroups = Boolean.parseBoolean(params.get("broadcast"));

			// get input rdd handle
			String groupsVar = params.get(Statement.GAGG_GROUPS);
			JavaPairRDD<MatrixIndexes, MatrixBlock> target = sec
				.getBinaryMatrixBlockRDDHandleForVariable(params.get(Statement.GAGG_TARGET));
			JavaPairRDD<MatrixIndexes, MatrixBlock> groups = broadcastGroups ? null : sec
				.getBinaryMatrixBlockRDDHandleForVariable(groupsVar);
			JavaPairRDD<MatrixIndexes, MatrixBlock> weights = null;

			DataCharacteristics mc1 = sec.getDataCharacteristics(params.get(Statement.GAGG_TARGET));
			DataCharacteristics mc2 = sec.getDataCharacteristics(groupsVar);
			if(mc1.dimsKnown() && mc2.dimsKnown() && (mc1.getRows() != mc2.getRows() || mc2.getCols() != 1)) {
				throw new DMLRuntimeException("Grouped Aggregate dimension mismatch between target and groups.");
			}
			DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());

			JavaPairRDD<MatrixIndexes, WeightedCell> groupWeightedCells = null;

			// Step 1: First extract groupWeightedCells from group, target and weights
			if(params.get(Statement.GAGG_WEIGHTS) != null) {
				weights = sec.getBinaryMatrixBlockRDDHandleForVariable(params.get(Statement.GAGG_WEIGHTS));

				DataCharacteristics mc3 = sec.getDataCharacteristics(params.get(Statement.GAGG_WEIGHTS));
				if(mc1.dimsKnown() && mc3.dimsKnown() &&
					(mc1.getRows() != mc3.getRows() || mc1.getCols() != mc3.getCols())) {
					throw new DMLRuntimeException(
						"Grouped Aggregate dimension mismatch between target, groups, and weights.");
				}

				groupWeightedCells = groups.join(target).join(weights).flatMapToPair(new ExtractGroupNWeights());
			}
			else // input vector or matrix
			{
				String ngroupsStr = params.get(Statement.GAGG_NUM_GROUPS);
				long ngroups = (ngroupsStr != null) ? (long) Double.parseDouble(ngroupsStr) : -1;

				// execute basic grouped aggregate (extract and preagg)
				if(broadcastGroups) {
					PartitionedBroadcast<MatrixBlock> pbm = sec.getBroadcastForVariable(groupsVar);
					groupWeightedCells = target
						.flatMapToPair(new ExtractGroupBroadcast(pbm, mc1.getBlocksize(), ngroups, _optr));
				}
				else { // general case

					// replicate groups if necessary
					if(mc1.getNumColBlocks() > 1) {
						groups = groups.flatMapToPair(new ReplicateVectorFunction(false, mc1.getNumColBlocks()));
					}

					groupWeightedCells = groups.join(target)
						.flatMapToPair(new ExtractGroupJoin(mc1.getBlocksize(), ngroups, _optr));
				}
			}

			// Step 2: Make sure we have blen required while creating <MatrixIndexes, MatrixCell>
			if(mc1.getBlocksize() == -1) {
				throw new DMLRuntimeException("The block sizes are not specified for grouped aggregate");
			}
			int blen = mc1.getBlocksize();

			// Step 3: Now perform grouped aggregate operation (either on combiner side or reducer side)
			JavaPairRDD<MatrixIndexes, MatrixCell> out = null;
			if(_optr instanceof CMOperator && ((CMOperator) _optr).isPartialAggregateOperator() ||
				_optr instanceof AggregateOperator) {
				out = groupWeightedCells.reduceByKey(new PerformGroupByAggInCombiner(_optr))
					.mapValues(new CreateMatrixCell(blen, _optr));
			}
			else {
				// Use groupby key because partial aggregation is not supported
				out = groupWeightedCells.groupByKey().mapValues(new PerformGroupByAggInReducer(_optr))
					.mapValues(new CreateMatrixCell(blen, _optr));
			}

			// Step 4: Set output characteristics and rdd handle
			setOutputCharacteristicsForGroupedAgg(mc1, mcOut, out);

			// store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), params.get(Statement.GAGG_TARGET));
			sec.addLineage(output.getName(), groupsVar, broadcastGroups);
			if(params.get(Statement.GAGG_WEIGHTS) != null) {
				sec.addLineageRDD(output.getName(), params.get(Statement.GAGG_WEIGHTS));
			}
		}
		else if(opcode.equalsIgnoreCase(Opcodes.RMEMPTY.toString())) {
			String rddInVar = params.get("target");
			String rddOffVar = params.get("offset");

			boolean rows = sec.getScalarInput(params.get("margin"), ValueType.STRING, true).getStringValue()
				.equals("rows");
			boolean emptyReturn = Boolean.parseBoolean(params.get("empty.return").toLowerCase());
			long maxDim = sec.getScalarInput(params.get("maxdim"), ValueType.FP64, false).getLongValue();
			boolean bRmEmptyBC = Boolean.parseBoolean(params.get("bRmEmptyBC"));
			DataCharacteristics mcIn = sec.getDataCharacteristics(rddInVar);

			if(maxDim > 0) // default case
			{
				// get input rdd handle
				JavaPairRDD<MatrixIndexes, MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable(rddInVar);
				JavaPairRDD<MatrixIndexes, MatrixBlock> off;
				PartitionedBroadcast<MatrixBlock> broadcastOff;
				long blen = mcIn.getBlocksize();
				long numRep = (long) Math.ceil(rows ? (double) mcIn.getCols() / blen : (double) mcIn.getRows() / blen);

				// execute remove empty rows/cols operation
				JavaPairRDD<MatrixIndexes, MatrixBlock> out;

				if(bRmEmptyBC) {
					broadcastOff = sec.getBroadcastForVariable(rddOffVar);
					// Broadcast offset vector
					out = in.flatMapToPair(new RDDRemoveEmptyFunctionInMem(rows, maxDim, blen, broadcastOff));
				}
				else {
					off = sec.getBinaryMatrixBlockRDDHandleForVariable(rddOffVar);
					out = in.join(off.flatMapToPair(new ReplicateVectorFunction(!rows, numRep)))
						.flatMapToPair(new RDDRemoveEmptyFunction(rows, maxDim, blen));
				}

				out = RDDAggregateUtils.mergeByKey(out, false);

				// store output rdd handle
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), rddInVar);
				if(bRmEmptyBC)
					sec.addLineageBroadcast(output.getName(), rddOffVar);
				else
					sec.addLineageRDD(output.getName(), rddOffVar);

				// update output statistics (required for correctness)
				DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
				mcOut.set(rows ? maxDim : mcIn.getRows(),
					rows ? mcIn.getCols() : maxDim,
					(int) blen,
					mcIn.getNonZeros());
			}
			else // special case: empty output (ensure valid dims)
			{
				int n = emptyReturn ? 1 : 0;
				MatrixBlock out = new MatrixBlock(rows ? n : (int) mcIn.getRows(), rows ? (int) mcIn.getCols() : n,
					true);
				sec.setMatrixOutput(output.getName(), out);
			}
		}
		else if(opcode.equalsIgnoreCase(Opcodes.CONTAINS.toString())) {
			
			JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = sec
				.getBinaryMatrixBlockRDDHandleForVariable(params.get("target"));
			
			Data pattern = ec.getVariable(params.get("pattern"));
			if( pattern == null ) //literal
				pattern = ScalarObjectFactory.createScalarObject(ValueType.FP64, params.get("pattern"));
			
			boolean ret = false;
			if( pattern.getDataType().isScalar() ) {
				double dpattern = ((ScalarObject)pattern).getDoubleValue();
				ret = in1.values() //num blocks containing pattern
					.map(new RDDContainsFunction(dpattern))
					.reduce((a,b) -> a+b) > 0;
			}
			else {
				PartitionedBroadcast<MatrixBlock> bc = sec.getBroadcastForVariable(params.get("pattern"));
				DataCharacteristics dc = sec.getDataCharacteristics(params.get("target"));
				ret = in1.flatMapToPair(new RDDContainsVectFunction(bc, dc.getBlocksize()))
					.reduceByKey((a,b) -> a+b)
					.values().reduce((a,b) -> Math.max(a,b)) == dc.getNumColBlocks();
			}
			
			// execute contains operation 
			ec.setScalarOutput(output.getName(), new BooleanObject(ret));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.REPLACE.toString())) {
			if(sec.isFrameObject(params.get("target"))){
				params.get("target");
				JavaPairRDD<Long, FrameBlock> in1 = sec.getFrameBinaryBlockRDDHandleForVariable(params.get("target"));
				DataCharacteristics mcIn = sec.getDataCharacteristics(params.get("target"));
				String pattern = params.get("pattern");
				String replacement = params.get("replacement");
				JavaPairRDD<Long, FrameBlock> out = in1.mapValues(new RDDFrameReplaceFunction(pattern, replacement));
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), params.get("target"));
				sec.getDataCharacteristics(output.getName()).set(mcIn.getRows(), mcIn.getCols(), mcIn.getBlocksize(), mcIn.getNonZeros());
			}
			else {
				JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = sec
					.getBinaryMatrixBlockRDDHandleForVariable(params.get("target"));
				DataCharacteristics mcIn = sec.getDataCharacteristics(params.get("target"));
	
				// execute replace operation
				double pattern = Double.parseDouble(params.get("pattern"));
				double replacement = Double.parseDouble(params.get("replacement"));
				JavaPairRDD<MatrixIndexes, MatrixBlock> out = in1.mapValues(new RDDReplaceFunction(pattern, replacement));
	
				// store output rdd handle
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), params.get("target"));
	
				// update output statistics (required for correctness)
				sec.getDataCharacteristics(output.getName()).set(mcIn.getRows(),
					mcIn.getCols(),
					mcIn.getBlocksize(),
					(pattern != 0 && replacement != 0) ? mcIn.getNonZeros() : -1);
			}
		}
		else if(opcode.equalsIgnoreCase(Opcodes.LOWERTRI.toString()) || opcode.equalsIgnoreCase(Opcodes.UPPERTRI.toString())) {
			JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = sec
				.getBinaryMatrixBlockRDDHandleForVariable(params.get("target"));
			DataCharacteristics mcIn = sec.getDataCharacteristics(params.get("target"));
			boolean lower = opcode.equalsIgnoreCase(Opcodes.LOWERTRI.toString());
			boolean diag = Boolean.parseBoolean(params.get("diag"));
			boolean values = Boolean.parseBoolean(params.get("values"));

			JavaPairRDD<MatrixIndexes, MatrixBlock> out = in1
				.mapPartitionsToPair(new RDDExtractTriangularFunction(lower, diag, values), true);

			// store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), params.get("target"));

			// update output statistics (required for correctness)
			sec.getDataCharacteristics(output.getName()).setDimension(mcIn.getRows(), mcIn.getCols());
		}
		else if(opcode.equalsIgnoreCase(Opcodes.REXPAND.toString())) {
			String rddInVar = params.get("target");

			// get input rdd handle
			JavaPairRDD<MatrixIndexes, MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable(rddInVar);
			DataCharacteristics mcIn = sec.getDataCharacteristics(rddInVar);
			
			// parse untyped parameters, w/ robust handling for 'max'
			String maxValName = params.get("max");
			long lmaxVal = maxValName.startsWith(Lop.SCALAR_VAR_NAME_PREFIX) ?
				ec.getScalarInput(maxValName, ValueType.FP64, false).getLongValue() :
				UtilFunctions.toLong(Double.parseDouble(maxValName));
			boolean dirRows = params.get("dir").equals("rows");
			boolean cast = Boolean.parseBoolean(params.get("cast"));
			boolean ignore = Boolean.parseBoolean(params.get("ignore"));
			long blen = mcIn.getBlocksize();

			// repartition input vector for higher degree of parallelism
			// (avoid scenarios where few input partitions create huge outputs)
			DataCharacteristics mcTmp = new MatrixCharacteristics(dirRows ? lmaxVal : mcIn.getRows(),
				dirRows ? mcIn.getRows() : lmaxVal, (int) blen, mcIn.getRows());
			int numParts = (int) Math.min(SparkUtils.getNumPreferredPartitions(mcTmp, in), mcIn.getNumBlocks());
			if(numParts > in.getNumPartitions() * 2)
				in = in.repartition(numParts);

			// execute rexpand rows/cols operation (no shuffle required because outputs are
			// block-aligned with the input, i.e., one input block generates n output blocks)
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = in
				.flatMapToPair(new RDDRExpandFunction(lmaxVal, dirRows, cast, ignore, blen));

			// store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddInVar);

			// update output statistics (required for correctness, nnz unknown due to cut-off)
			DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
			mcOut.set(dirRows ? lmaxVal : mcIn.getRows(), dirRows ? mcIn.getRows() : lmaxVal, (int) blen, -1);
			mcOut.setNonZerosBound(mcIn.getRows());

			// post-processing to obtain sparsity of ultra-sparse outputs
			SparkUtils.postprocessUltraSparseOutput(sec.getMatrixObject(output), mcOut);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TOKENIZE.toString())) {
			// get input RDD data
			FrameObject fo = sec.getFrameObject(params.get("target"));
			JavaPairRDD<Long, FrameBlock> in = (JavaPairRDD<Long, FrameBlock>) sec.getRDDHandleForFrameObject(fo,
				FileFormat.BINARY);
			DataCharacteristics mc = sec.getDataCharacteristics(params.get("target"));

			// construct tokenizer and tokenize text
			Tokenizer tokenizer = TokenizerFactory.createTokenizer(params.get("spec"),
				Integer.parseInt(params.get("max_tokens")));
			JavaPairRDD<Long, FrameBlock> out = in.mapToPair(new RDDTokenizeFunction(tokenizer, mc.getBlocksize()));

			// set output and maintain lineage/output characteristics
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), params.get("target"));

			// get max tokens for row upper bound
			long numRows = tokenizer.getMaxNumRows((int)mc.getRows());
			long numCols = tokenizer.getNumCols();

			sec.getDataCharacteristics(output.getName()).set(numRows, numCols, mc.getBlocksize());
			sec.getFrameObject(output.getName()).setSchema(tokenizer.getSchema());
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TRANSFORMAPPLY.toString())) {
			// get input RDD and meta data
			FrameObject fo = sec.getFrameObject(params.get("target"));
			JavaPairRDD<Long, FrameBlock> in = (JavaPairRDD<Long, FrameBlock>) sec.getRDDHandleForFrameObject(fo,
				FileFormat.BINARY);
			FrameBlock meta = sec.getFrameInput(params.get("meta"));
			MatrixBlock embeddings = params.get("embedding") != null ? ec.getMatrixInput(params.get("embedding")) : null;

			DataCharacteristics mcIn = sec.getDataCharacteristics(params.get("target"));
			DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
			String[] colnames = !TfMetaUtils.isIDSpec(params.get("spec")) ? in.lookup(1L).get(0)
				.getColumnNames() : null;

			// compute omit offset map for block shifts
			TfOffsetMap omap = null;
			if(TfMetaUtils.containsOmitSpec(params.get("spec"), colnames)) {
				omap = new TfOffsetMap(SparkUtils.toIndexedLong(
					in.mapToPair(new RDDTransformApplyOffsetFunction(params.get("spec"), colnames)).collect()));
			}

			// create encoder broadcast (avoiding replication per task)
			MultiColumnEncoder encoder = EncoderFactory
				.createEncoder(params.get("spec"), colnames, fo.getSchema(), (int) fo.getNumColumns(), meta, embeddings);
			encoder.updateAllDCEncoders();
			mcOut.setDimension(mcIn.getRows() - ((omap != null) ? omap.getNumRmRows() : 0), encoder.getNumOutCols());

			long t0 = System.nanoTime();
			Broadcast<MultiColumnEncoder> bmeta = sec.getSparkContext().broadcast(encoder);
			Broadcast<TfOffsetMap> bomap = (omap != null) ? sec.getSparkContext().broadcast(omap) : null;
			if (DMLScript.STATISTICS) {
				SparkStatistics.accBroadCastTime(System.nanoTime() - t0);
				SparkStatistics.incBroadcastCount(1);
			}

			// execute transform apply
			JavaPairRDD<MatrixIndexes, MatrixBlock> out;
			Tuple2<Boolean, Integer> aligned = FrameRDDAggregateUtils.checkRowAlignment(in, -1);
			// NOTE: currently disabled for LegacyEncoders, because OMIT probably results in not aligned
			// blocks and for IMPUTE was an inaccuracy for the "testHomesImputeColnamesSparkCSV" test case.

			// Error in test case: Expected: 8.150349617004395 vs actual: 8.15035 at 0 8 (expected is calculated from transform encode,
			// which currently always uses the else branch: either inaccuracy must come from serialisation of
			// matrixblock or from binaryBlockToBinaryBlock reblock
			if(aligned._1 && mcOut.getCols() <= aligned._2 && !encoder.hasLegacyEncoder() /*&& containsWE*/) {
				//Blocks are aligned & #Col is below Block length (necessary for matrix-matrix reblock)
				JavaPairRDD<Long, MatrixBlock> tmp = in.mapToPair(new RDDTransformApplyFunction2(bmeta, bomap));
				mcIn.setBlocksize(aligned._2);
				mcIn.setDimension(mcIn.getRows(), mcOut.getCols());
				JavaPairRDD<MatrixIndexes, MatrixBlock> tmp2 = tmp.mapToPair((PairFunction<Tuple2<Long, MatrixBlock>, MatrixIndexes, MatrixBlock>) in12 ->
						new Tuple2<>(new MatrixIndexes(UtilFunctions.computeBlockIndex(in12._1, aligned._2),1), in12._2));
				out = RDDConverterUtils.binaryBlockToBinaryBlock(tmp2, mcIn, mcOut);
				//out = RDDConverterUtils.matrixBlockToAlignedMatrixBlock(tmp, mcOut, mcOut);
			} else {
				JavaPairRDD<Long, FrameBlock> tmp = in.mapToPair(new RDDTransformApplyFunction(bmeta, bomap));
				out = FrameRDDConverterUtils.binaryBlockToMatrixBlock(tmp, mcOut, mcOut);
			}

			// set output and maintain lineage/output characteristics
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), params.get("target"));
			ec.releaseFrameInput(params.get("meta"));
			if(params.get("embedding") != null)
				ec.releaseMatrixInput(params.get("embedding"));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TRANSFORMDECODE.toString())) {
			// get input RDD and meta data
			JavaPairRDD<MatrixIndexes, MatrixBlock> in = sec
				.getBinaryMatrixBlockRDDHandleForVariable(params.get("target"));
			DataCharacteristics mc = sec.getDataCharacteristics(params.get("target"));
			FrameBlock meta = sec.getFrameInput(params.get("meta"));
			String[] colnames = meta.getColumnNames();

			// reblock if necessary (clen > blen)
			if(mc.getCols() > mc.getNumColBlocks()) {
				in = in.mapToPair(new RDDTransformDecodeExpandFunction((int) mc.getCols(), mc.getBlocksize()));
				in = RDDAggregateUtils.mergeByKey(in, false);
			}

			// construct decoder and decode individual matrix blocks
			Decoder decoder = DecoderFactory.createDecoder(params.get("spec"), colnames, null, meta);
			JavaPairRDD<Long, FrameBlock> out = in
				.mapToPair(new RDDTransformDecodeFunction(decoder, mc.getBlocksize()));

			// set output and maintain lineage/output characteristics
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), params.get("target"));
			ec.releaseFrameInput(params.get("meta"));
			sec.getDataCharacteristics(output.getName()).set(mc.getRows(), meta.getNumColumns(), mc.getBlocksize(), -1);
			sec.getFrameObject(output.getName()).setSchema(decoder.getSchema());
		}
		else {
			throw new DMLRuntimeException("Unknown parameterized builtin opcode: " + opcode);
		}
	}
	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		String opcode = getOpcode();
		if(opcode.equalsIgnoreCase(Opcodes.REPLACE.toString())) {
			CPOperand target = getTargetOperand();
			CPOperand pattern = getFP64Literal("pattern");
			CPOperand replace = getFP64Literal("replacement");
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, pattern, replace)));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.RMEMPTY.toString())) {
			CPOperand target = getTargetOperand();
			String off = params.get("offset");
			CPOperand offset = new CPOperand(off, ValueType.FP64, Types.DataType.MATRIX);
			CPOperand margin = getStringLiteral("margin");
			CPOperand emptyReturn = getBoolLiteral("empty.return");
			CPOperand maxDim = getLiteral("maxdim", ValueType.FP64);
			CPOperand bRmEmptyBC = getBoolLiteral("bRmEmptyBC");
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, offset, margin,
				emptyReturn, maxDim, bRmEmptyBC)));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TRANSFORMDECODE.toString()) || opcode.equalsIgnoreCase(Opcodes.TRANSFORMAPPLY.toString())) {
			CPOperand target = new CPOperand(params.get("target"), ValueType.FP64, Types.DataType.FRAME);
			CPOperand meta = getLiteral("meta", ValueType.UNKNOWN, Types.DataType.FRAME);
			CPOperand spec = getStringLiteral("spec");
			//FIXME: Taking only spec file name as a literal leads to wrong reuse
			//TODO: Add Embedding to the lineage item
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, meta, spec)));
		}
		if(opcode.equalsIgnoreCase(Opcodes.CONTAINS.toString())) {
			CPOperand target = getTargetOperand();
			CPOperand pattern = getFP64Literal("pattern");
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, pattern)));
		}
		else {
			// NOTE: for now, we cannot have a generic fall through path, because the
			// data and value types of parmeters are not compiled into the instruction
			throw new DMLRuntimeException("Unsupported lineage tracing for: " + opcode);
		}
	}

	private CPOperand getTargetOperand() {
		return new CPOperand(params.get("target"), ValueType.FP64, Types.DataType.MATRIX);
	}

	private CPOperand getFP64Literal(String name) {
		return getLiteral(name, ValueType.FP64);
	}

	private CPOperand getStringLiteral(String name) {
		return getLiteral(name, ValueType.STRING);
	}

	private CPOperand getLiteral(String name, ValueType vt) {
		return new CPOperand(params.get(name), vt, Types.DataType.SCALAR, true);
	}

	private CPOperand getLiteral(String name, ValueType vt, Types.DataType dt) {
		return new CPOperand(params.get(name), vt, dt);
	}

	private CPOperand getBoolLiteral(String name) {
		return getLiteral(name, ValueType.BOOLEAN);
	}

	public HashMap<String, String> getParameterMap() {
		return params;
	}

	public static class RDDReplaceFunction implements Function<MatrixBlock, MatrixBlock> {
		private static final long serialVersionUID = 6576713401901671659L;
		private final double _pattern;
		private final double _replacement;

		public RDDReplaceFunction(double pattern, double replacement) {
			_pattern = pattern;
			_replacement = replacement;
		}

		@Override
		public MatrixBlock call(MatrixBlock arg0) {
			return arg0.replaceOperations(new MatrixBlock(), _pattern, _replacement);
		}
	}
	
	public static class RDDContainsFunction implements Function<MatrixBlock, Double> {
		private static final long serialVersionUID = 6576713401901671659L;
		private final double _pattern;

		public RDDContainsFunction(double pattern) {
			_pattern = pattern;
		}

		@Override
		public Double call(MatrixBlock arg0) {
			return arg0.containsValue(_pattern) ? 1d : 0d;
		}
	}
	
	public static class RDDContainsVectFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, Long, Integer>
	{
		private static final long serialVersionUID = 2228503788469700742L;
		private final PartitionedBroadcast<MatrixBlock> _pbcPattern;
		private final int _blocksize;
		
		public RDDContainsVectFunction(PartitionedBroadcast<MatrixBlock> bc, int blocksize) {
			_pbcPattern = bc;
			_blocksize = blocksize;
		}

		@Override
		public Iterator<Tuple2<Long, Integer>> call(Tuple2<MatrixIndexes, MatrixBlock> input) throws Exception {
			MatrixIndexes ix = input._1();
			MatrixBlock pattern = _pbcPattern.getBlock(1, (int)ix.getColumnIndex());
			List<Integer> tmp = input._2().containsVector(pattern, false);
			
			List<Tuple2<Long,Integer>> ret = new ArrayList<>(); 
			ret.add(new Tuple2<>(ix.getRowIndex()*_blocksize, 0)); //ensure non-empty RDD
			for(int rix : tmp)
				ret.add(new Tuple2<>(UtilFunctions.computeCellIndex(ix.getRowIndex(), _blocksize, rix), 1));
			return ret.iterator();
		}
	}

	public static class RDDFrameReplaceFunction implements Function<FrameBlock, FrameBlock>{
		private static final long serialVersionUID = 6576713401901671660L;
		private final String _pattern;
		private final String _replacement;

		public RDDFrameReplaceFunction(String pattern, String replacement){
			_pattern = pattern;
			_replacement = replacement;
		}

		@Override 
		public FrameBlock call(FrameBlock arg0){
			return arg0.replaceOperations(_pattern, _replacement);
		}
	}

	private static class RDDExtractTriangularFunction
		implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock>>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = 2754868819184155702L;
		private final boolean _lower, _diag, _values;

		public RDDExtractTriangularFunction(boolean lower, boolean diag, boolean values) {
			_lower = lower;
			_diag = diag;
			_values = values;
		}

		@Override
		public LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>> call(
			Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0) {
			return new ExtractTriangularIterator(arg0);
		}

		private class ExtractTriangularIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>> {
			public ExtractTriangularIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}

			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg) {
				MatrixIndexes ix = arg._1();
				MatrixBlock mb = arg._2();

				// handle cases of pass-through and reset block
				if((_lower && ix.getRowIndex() > ix.getColumnIndex()) ||
					(!_lower && ix.getRowIndex() < ix.getColumnIndex())) {
					return _values ? arg : new Tuple2<>(ix, new MatrixBlock(mb.getNumRows(), mb.getNumColumns(), 1d));
				}

				// handle cases of empty blocks
				if((_lower && ix.getRowIndex() < ix.getColumnIndex()) ||
					(!_lower && ix.getRowIndex() > ix.getColumnIndex())) {
					return new Tuple2<>(ix, new MatrixBlock(mb.getNumRows(), mb.getNumColumns(), true));
				}

				// extract triangular blocks for blocks on diagonal
				assert (ix.getRowIndex() == ix.getColumnIndex());
				return new Tuple2<>(ix, mb.extractTriangular(new MatrixBlock(), _lower, _diag, _values));
			}
		}
	}

	public static class RDDRemoveEmptyFunction implements
		PairFlatMapFunction<Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = 4906304771183325289L;

		private final boolean _rmRows;
		private final long _len;
		private final long _blen;

		public RDDRemoveEmptyFunction(boolean rmRows, long len, long blen) {
			_rmRows = rmRows;
			_len = len;
			_blen = blen;
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(
			Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> arg0) throws Exception {
			// prepare inputs (for internal api compatibility)
			IndexedMatrixValue data = SparkUtils.toIndexedMatrixBlock(arg0._1(), arg0._2()._1());
			IndexedMatrixValue offsets = SparkUtils.toIndexedMatrixBlock(arg0._1(), arg0._2()._2());

			// execute remove empty operations
			ArrayList<IndexedMatrixValue> out = new ArrayList<>();
			LibMatrixReorg.rmempty(data, offsets, _rmRows, _len, _blen, out);

			// prepare and return outputs
			return SparkUtils.fromIndexedMatrixBlock(out).iterator();
		}
	}

	public static class RDDRemoveEmptyFunctionInMem
		implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = 4906304771183325289L;

		private final boolean _rmRows;
		private final long _len;
		private final long _blen;

		private PartitionedBroadcast<MatrixBlock> _off = null;

		public RDDRemoveEmptyFunctionInMem(boolean rmRows, long len, long blen, PartitionedBroadcast<MatrixBlock> off) {
			_rmRows = rmRows;
			_len = len;
			_blen = blen;
			_off = off;
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception {
			// prepare inputs (for internal api compatibility)
			IndexedMatrixValue data = SparkUtils.toIndexedMatrixBlock(arg0._1(), arg0._2());
			IndexedMatrixValue offsets = _rmRows ? SparkUtils.toIndexedMatrixBlock(arg0._1(),
				_off.getBlock((int) arg0._1().getRowIndex(), 1)) : SparkUtils.toIndexedMatrixBlock(arg0._1(),
					_off.getBlock(1, (int) arg0._1().getColumnIndex()));

			// execute remove empty operations
			ArrayList<IndexedMatrixValue> out = new ArrayList<>();
			LibMatrixReorg.rmempty(data, offsets, _rmRows, _len, _blen, out);

			// prepare and return outputs
			return SparkUtils.fromIndexedMatrixBlock(out).iterator();
		}
	}

	public static class RDDRExpandFunction
		implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = -6153643261956222601L;

		private final long _maxVal;
		private final boolean _dirRows;
		private final boolean _cast;
		private final boolean _ignore;
		private final long _blen;

		public RDDRExpandFunction(long maxVal, boolean dirRows, boolean cast, boolean ignore, long blen) {
			_maxVal = maxVal;
			_dirRows = dirRows;
			_cast = cast;
			_ignore = ignore;
			_blen = blen;
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception {
			// prepare inputs (for internal api compatibility)
			IndexedMatrixValue data = SparkUtils.toIndexedMatrixBlock(arg0._1(), arg0._2());

			// execute rexpand operations
			ArrayList<IndexedMatrixValue> out = new ArrayList<>();
			LibMatrixReorg.rexpand(data, _maxVal, _dirRows, _cast, _ignore, _blen, out);

			// prepare and return outputs
			return SparkUtils.fromIndexedMatrixBlock(out).iterator();
		}
	}

	public static class RDDMapGroupedAggFunction
		implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = 6795402640178679851L;

		private PartitionedBroadcast<MatrixBlock> _pbm = null;
		private Operator _op = null;
		private int _ngroups = -1;
		private int _blen = -1;

		public RDDMapGroupedAggFunction(PartitionedBroadcast<MatrixBlock> pbm, Operator op, int ngroups, int blen) {
			_pbm = pbm;
			_op = op;
			_ngroups = ngroups;
			_blen = blen;
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception {
			// get all inputs
			MatrixIndexes ix = arg0._1();
			MatrixBlock target = arg0._2();
			MatrixBlock groups = _pbm.getBlock((int) ix.getRowIndex(), 1);

			// execute map grouped aggregate operations
			IndexedMatrixValue in1 = SparkUtils.toIndexedMatrixBlock(ix, target);
			ArrayList<IndexedMatrixValue> outlist = new ArrayList<>();
			OperationsOnMatrixValues.performMapGroupedAggregate(_op, in1, groups, _ngroups, _blen, outlist);

			// output all result blocks
			return SparkUtils.fromIndexedMatrixBlock(outlist).iterator();
		}
	}

	/**
	 * Similar to RDDMapGroupedAggFunction but single output block.
	 */
	public static class RDDMapGroupedAggFunction2 implements Function<Tuple2<MatrixIndexes, MatrixBlock>, MatrixBlock> {
		private static final long serialVersionUID = -6820599604299797661L;

		private PartitionedBroadcast<MatrixBlock> _pbm = null;
		private Operator _op = null;
		private int _ngroups = -1;

		public RDDMapGroupedAggFunction2(PartitionedBroadcast<MatrixBlock> pbm, Operator op, int ngroups) {
			_pbm = pbm;
			_op = op;
			_ngroups = ngroups;
		}

		@Override
		public MatrixBlock call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			// get all inputs
			MatrixIndexes ix = arg0._1();
			MatrixBlock target = arg0._2();
			MatrixBlock groups = _pbm.getBlock((int) ix.getRowIndex(), 1);

			// execute map grouped aggregate operations
			return groups.groupedAggOperations(target, null, new MatrixBlock(), _ngroups, _op);
		}
	}

	public static class CreateMatrixCell implements Function<WeightedCell, MatrixCell> {
		private static final long serialVersionUID = -5783727852453040737L;

		int blen;
		Operator op;

		public CreateMatrixCell(int blen, Operator op) {
			this.blen = blen;
			this.op = op;
		}

		@Override
		public MatrixCell call(WeightedCell kv) throws Exception {
			double val = -1;
			if(op instanceof CMOperator) {
				AggregateOperationTypes agg = ((CMOperator) op).aggOpType;
				switch(agg) {
					case COUNT:
						val = kv.getWeight();
						break;
					case MEAN:
						val = kv.getValue();
						break;
					case MIN:
						val = kv.getValue();
						break;
					case MAX:
						val = kv.getValue();
						break;
					case CM2:
						val = kv.getValue() / kv.getWeight();
						break;
					case CM3:
						val = kv.getValue() / kv.getWeight();
						break;
					case CM4:
						val = kv.getValue() / kv.getWeight();
						break;
					case VARIANCE:
						val = kv.getValue() / kv.getWeight();
						break;
					default:
						throw new DMLRuntimeException("Invalid aggreagte in CM_CV_Object: " + agg);
				}
			}
			else {
				// avoid division by 0
				val = kv.getValue() / kv.getWeight();
			}

			return new MatrixCell(val);
		}
	}

	public static class RDDTokenizeFunction implements PairFunction<Tuple2<Long, FrameBlock>, Long, FrameBlock> {
		private static final long serialVersionUID = -8788298032616522019L;

		private Tokenizer _tokenizer = null;
		
		public RDDTokenizeFunction(Tokenizer tokenizer, int blen) {
			_tokenizer = tokenizer;
		}

		@Override
		public Tuple2<Long, FrameBlock> call(Tuple2<Long, FrameBlock> in) throws Exception {
			long key = in._1();
			FrameBlock blk = in._2();

			FrameBlock fbout = _tokenizer.tokenize(blk, OptimizerUtils.getTokenizeNumThreads());
			return new Tuple2<>(key, fbout);
		}
	}

	public static class RDDTransformApplyFunction implements PairFunction<Tuple2<Long, FrameBlock>, Long, FrameBlock> {
		private static final long serialVersionUID = 5759813006068230916L;

		private Broadcast<MultiColumnEncoder> _bencoder = null;
		private Broadcast<TfOffsetMap> _omap = null;

		public RDDTransformApplyFunction(Broadcast<MultiColumnEncoder> bencoder, Broadcast<TfOffsetMap> omap) {
			_bencoder = bencoder;
			_omap = omap;
		}

		@Override
		public Tuple2<Long, FrameBlock> call(Tuple2<Long, FrameBlock> in) throws Exception {
			long key = in._1();
			FrameBlock blk = in._2();

			// execute block transform apply
			MultiColumnEncoder encoder = _bencoder.getValue();
			// we need to create a copy of the encoder since the bag of word encoder stores frameblock specific state
			// which would be overwritten when multiple blocks are located on a executor
			// to avoid this, we need to create a shallow copy of the MCEncoder, where we only instantiate new bow
			// encoders objects with the frameblock specific fields and shallow copy the other fields (like meta)
			// other encoders are reused and not newly instantiated
			if(!encoder.getColumnEncoders(ColumnEncoderBagOfWords.class).isEmpty())
				encoder = new MultiColumnEncoder(encoder); // create copy
			MatrixBlock tmp = encoder.apply(blk);
			// remap keys
			if(_omap != null) {
				key = _omap.getValue().getOffset(key);
			}

			// convert to frameblock to reuse frame-matrix reblock
			return new Tuple2<>(key, DataConverter.convertToFrameBlock(tmp));
		}
	}



	public static class RDDTransformApplyOffsetFunction implements PairFunction<Tuple2<Long, FrameBlock>, Long, Long> {
		private static final long serialVersionUID = 3450977356721057440L;

		private int[] _omitColList = null;

		public RDDTransformApplyOffsetFunction(String spec, String[] colnames) {
			try {
				_omitColList = TfMetaUtils.parseJsonIDList(spec, colnames, TfMethod.OMIT.toString());
			}
			catch(DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}

		@Override
		public Tuple2<Long, Long> call(Tuple2<Long, FrameBlock> in) throws Exception {
			long key = in._1();
			long rmRows = 0;

			FrameBlock blk = in._2();

			for(int i = 0; i < blk.getNumRows(); i++) {
				boolean valid = true;
				for(int j = 0; j < _omitColList.length; j++) {
					int colID = _omitColList[j];
					Object val = blk.get(i, colID - 1);
					valid &= !(val == null ||
						(blk.getSchema()[colID - 1] == ValueType.STRING && val.toString().isEmpty()));
				}
				rmRows += valid ? 0 : 1;
			}

			return new Tuple2<>(key, rmRows);
		}
	}

	public static class RDDTransformApplyFunction2 implements PairFunction<Tuple2<Long, FrameBlock>, Long, MatrixBlock> {
		private static final long serialVersionUID = 5759813006068230916L;

		private Broadcast<MultiColumnEncoder> _bencoder = null;
		private Broadcast<TfOffsetMap> _omap = null;

		public RDDTransformApplyFunction2(Broadcast<MultiColumnEncoder> bencoder, Broadcast<TfOffsetMap> omap) {
			_bencoder = bencoder;
			_omap = omap;
		}

		@Override
		public Tuple2<Long, MatrixBlock> call(Tuple2<Long, FrameBlock> in) throws Exception {
			long key = in._1();
			FrameBlock blk = in._2();

			// execute block transform apply
			MultiColumnEncoder encoder = _bencoder.getValue();
			MatrixBlock tmp = encoder.apply(blk);
			// remap keys
			if(_omap != null) {
				key = _omap.getValue().getOffset(key);
			}

			// convert to frameblock to reuse frame-matrix reblock
			return new Tuple2<>(key, tmp);
		}
	}

	public static class RDDTransformDecodeFunction
		implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, Long, FrameBlock> {
		private static final long serialVersionUID = -4797324742568170756L;

		private Decoder _decoder = null;
		private int _blen = -1;

		public RDDTransformDecodeFunction(Decoder decoder, int blen) {
			_decoder = decoder;
			_blen = blen;
		}

		@Override
		public Tuple2<Long, FrameBlock> call(Tuple2<MatrixIndexes, MatrixBlock> in) throws Exception {
			long rix = UtilFunctions.computeCellIndex(in._1().getRowIndex(), _blen, 0);
			FrameBlock fbout = _decoder.decode(in._2(), new FrameBlock(_decoder.getSchema()));
			fbout.setColumnNames(Arrays.copyOfRange(_decoder.getColnames(), 0, fbout.getNumColumns()));
			return new Tuple2<>(rix, fbout);
		}
	}

	public static class RDDTransformDecodeExpandFunction
		implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = -8187400248076127598L;

		private int _clen = -1;
		private int _blen = -1;

		public RDDTransformDecodeExpandFunction(int clen, int blen) {
			_clen = clen;
			_blen = blen;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> in) throws Exception {
			MatrixIndexes inIx = in._1();
			MatrixBlock inBlk = in._2();

			// construct expanded block via leftindexing
			int cl = (int) UtilFunctions.computeCellIndex(inIx.getColumnIndex(), _blen, 0) - 1;
			int cu = (int) UtilFunctions.computeCellIndex(inIx.getColumnIndex(), _blen, inBlk.getNumColumns() - 1) - 1;
			MatrixBlock out = new MatrixBlock(inBlk.getNumRows(), _clen, false);
			out = out.leftIndexingOperations(inBlk, 0, inBlk.getNumRows() - 1, cl, cu, null, UpdateType.INPLACE_PINNED);

			return new Tuple2<>(new MatrixIndexes(inIx.getRowIndex(), 1), out);
		}
	}

	public void setOutputCharacteristicsForGroupedAgg(DataCharacteristics mc1, DataCharacteristics mcOut,
		JavaPairRDD<MatrixIndexes, MatrixCell> out) {
		if(!mcOut.dimsKnown()) {
			if(!mc1.dimsKnown()) {
				throw new DMLRuntimeException("The output dimensions are not specified for grouped aggregate");
			}

			if(params.get(Statement.GAGG_NUM_GROUPS) != null) {
				int ngroups = (int) Double.parseDouble(params.get(Statement.GAGG_NUM_GROUPS));
				mcOut.set(ngroups, mc1.getCols(), -1, -1); // grouped aggregate with cell output
			}
			else {
				out = SparkUtils.cacheBinaryCellRDD(out);
				mcOut.set(SparkUtils.computeDataCharacteristics(out));
				mcOut.setBlocksize(-1); // grouped aggregate with cell output
			}
		}
	}
}
