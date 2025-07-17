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

package org.apache.sysds.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.ParameterizedBuiltinFunctionExpression;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.TensorObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.functionobjects.ParameterizedBuiltin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.decode.ColumnDecoder;
import org.apache.sysds.runtime.transform.decode.ColumnDecoderFactory;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.transform.tokenize.Tokenizer;
import org.apache.sysds.runtime.transform.tokenize.TokenizerFactory;
import org.apache.sysds.runtime.util.AutoDiff;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class ParameterizedBuiltinCPInstruction extends ComputationCPInstruction {
	private static final Log LOG = LogFactory.getLog(ParameterizedBuiltinCPInstruction.class.getName());
	private static final int TOSTRING_MAXROWS = 100;
	private static final int TOSTRING_MAXCOLS = 100;
	private static final int TOSTRING_DECIMAL = 3;
	private static final boolean TOSTRING_SPARSE = false;
	private static final String TOSTRING_SEPARATOR = " ";
	private static final String TOSTRING_LINESEPARATOR = "\n";

	protected final LinkedHashMap<String, String> params;

	protected ParameterizedBuiltinCPInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out,
		String opcode, String istr) {
		super(CPType.ParameterizedBuiltin, op, null, null, out, opcode, istr);
		params = paramsMap;
	}

	public HashMap<String, String> getParameterMap() {
		return params;
	}

	public String getParam(String key) {
		return getParameterMap().get(key);
	}

	public static LinkedHashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		LinkedHashMap<String, String> paramMap = new LinkedHashMap<>();
		// all parameters are of form <name=value>
		String[] parts;
		for(int i = 1; i <= params.length - 2; i++) {
			parts = params[i].split(Lop.NAME_VALUE_SEPARATOR);
			paramMap.put(parts[0], parts[1]);
		}

		return paramMap;
	}

	public static ParameterizedBuiltinCPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand(parts[parts.length - 1]);

		// process remaining parts and build a hash map
		LinkedHashMap<String, String> paramsMap = constructParameterMap(parts);

		// determine the appropriate value function
		ValueFunction func = null;
		if(opcode.equalsIgnoreCase(Opcodes.CDF.toString())) {
			if(paramsMap.get("dist") == null)
				throw new DMLRuntimeException("Invalid distribution: " + str);
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode, paramsMap.get("dist"));
			// Determine appropriate Function Object based on opcode
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.INVCDF.toString())) {
			if(paramsMap.get("dist") == null)
				throw new DMLRuntimeException("Invalid distribution: " + str);
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode, paramsMap.get("dist"));
			// Determine appropriate Function Object based on opcode
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.GROUPEDAGG.toString())) {
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
			return new ParameterizedBuiltinCPInstruction(op, paramsMap, out, opcode, str);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.RMEMPTY.toString()) || opcode.equalsIgnoreCase(Opcodes.REPLACE.toString()) ||
			opcode.equalsIgnoreCase(Opcodes.REXPAND.toString()) || opcode.equalsIgnoreCase(Opcodes.LOWERTRI.toString()) ||
			opcode.equalsIgnoreCase(Opcodes.UPPERTRI.toString()) ) {
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if(opcode.equals(Opcodes.TRANSFORMAPPLY.toString()) || opcode.equals(Opcodes.TRANSFORMDECODE.toString())
			|| opcode.equalsIgnoreCase(Opcodes.CONTAINS.toString()) || opcode.equals(Opcodes.TRANSFORMCOLMAP.toString())
			|| opcode.equals(Opcodes.TRANSFORMMETA.toString()) || opcode.equals(Opcodes.TOKENIZE.toString())
			|| opcode.equals(Opcodes.TOSTRING.toString()) || opcode.equals(Opcodes.NVLIST.toString()) || opcode.equals(Opcodes.AUTODIFF.toString())) {
			return new ParameterizedBuiltinCPInstruction(null, paramsMap, out, opcode, str);
		}
		else if(Opcodes.PARAMSERV.toString().equals(opcode)) {
			return new ParamservBuiltinCPInstruction(null, paramsMap, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode (" + opcode + ") for ParameterizedBuiltin Instruction.");
		}

	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		ScalarObject sores = null;
		if(opcode.equalsIgnoreCase(Opcodes.CDF.toString())) {
			SimpleOperator op = (SimpleOperator) _optr;
			double result = op.fn.execute(params);
			sores = new DoubleObject(result);
			ec.setScalarOutput(output.getName(), sores);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.INVCDF.toString())) {
			SimpleOperator op = (SimpleOperator) _optr;
			double result = op.fn.execute(params);
			sores = new DoubleObject(result);
			ec.setScalarOutput(output.getName(), sores);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.AUTODIFF.toString()))
		{
			ArrayList<Data> lineage = (ArrayList<Data>) ec.getListObject(params.get("lineage")).getData();
			MatrixObject mo = ec.getMatrixObject(params.get("output"));
			ListObject diffs = AutoDiff.getBackward(mo, lineage, ExecutionContextFactory.createContext());
			ec.setVariable(output.getName(), diffs);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.GROUPEDAGG.toString())) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get(Statement.GAGG_TARGET));
			MatrixBlock groups = ec.getMatrixInput(params.get(Statement.GAGG_GROUPS));
			MatrixBlock weights = null;
			if(params.get(Statement.GAGG_WEIGHTS) != null)
				weights = ec.getMatrixInput(params.get(Statement.GAGG_WEIGHTS));

			int ngroups = -1;
			if(params.get(Statement.GAGG_NUM_GROUPS) != null) {
				ngroups = (int) Double.parseDouble(params.get(Statement.GAGG_NUM_GROUPS));
			}

			// compute the result
			int k = Integer.parseInt(params.get("k")); // num threads
			MatrixBlock soresBlock = groups.groupedAggOperations(target, weights, new MatrixBlock(), ngroups, _optr, k);

			ec.setMatrixOutput(output.getName(), soresBlock);
			// release locks
			target = groups = weights = null;
			ec.releaseMatrixInput(params.get(Statement.GAGG_TARGET));
			ec.releaseMatrixInput(params.get(Statement.GAGG_GROUPS));
			if(params.get(Statement.GAGG_WEIGHTS) != null)
				ec.releaseMatrixInput(params.get(Statement.GAGG_WEIGHTS));

		}
		else if(opcode.equalsIgnoreCase(Opcodes.RMEMPTY.toString())) {
			String margin = params.get("margin");
			if(!(margin.equals("rows") || margin.equals("cols")))
				throw new DMLRuntimeException("Unspupported margin identifier '" + margin + "'.");
			if(ec.isFrameObject(params.get("target"))) {
				FrameBlock target = ec.getFrameInput(params.get("target"));
				MatrixBlock select = params.containsKey("select") ? ec.getMatrixInput(params.get("select")) : null;

				boolean emptyReturn = Boolean.parseBoolean(params.get("empty.return").toLowerCase());
				FrameBlock soresBlock = target.removeEmptyOperations(margin.equals("rows"), emptyReturn, select);
				ec.setFrameOutput(output.getName(), soresBlock);
				ec.releaseFrameInput(params.get("target"));
				if(params.containsKey("select"))
					ec.releaseMatrixInput(params.get("select"));
			} else {
				// acquire locks
				MatrixBlock target = ec.getMatrixInput(params.get("target"));
				MatrixBlock select = params.containsKey("select") ? ec.getMatrixInput(params.get("select")) : null;

				// compute the result
				boolean emptyReturn = Boolean.parseBoolean(params.get("empty.return").toLowerCase());
				MatrixBlock ret = target.removeEmptyOperations(new MatrixBlock(), margin.equals("rows"), emptyReturn, select);

				// release locks
				if( target == ret ) //short-circuit (avoid buffer pool pollution)
					ec.setVariable(output.getName(), ec.getVariable(params.get("target")));
				else
					ec.setMatrixOutput(output.getName(), ret);
				ec.releaseMatrixInput(params.get("target"));
				if(params.containsKey("select"))
					ec.releaseMatrixInput(params.get("select"));
			}
		}
		else if(opcode.equalsIgnoreCase(Opcodes.CONTAINS.toString())) {
			String varName = params.get("target");
			int k = Integer.parseInt(params.get("k")); //num threads
			MatrixBlock target = ec.getMatrixInput(varName);
			Data pattern = ec.getVariable(params.get("pattern"));
			if( pattern == null ) //literal
				pattern = ScalarObjectFactory.createScalarObject(ValueType.FP64, params.get("pattern"));
			boolean ret = pattern.getDataType().isScalar() ?
				target.containsValue(((ScalarObject)pattern).getDoubleValue(), k) : 
				(target.containsVector(((MatrixObject)pattern).acquireRead(), true).size()>0);
			ec.releaseMatrixInput(varName);
			if(!pattern.getDataType().isScalar())
				ec.releaseMatrixInput(params.get("pattern"));
			ec.setScalarOutput(output.getName(), new BooleanObject(ret));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.REPLACE.toString())) {
			if(ec.isFrameObject(params.get("target"))){
				FrameBlock target = ec.getFrameInput(params.get("target"));
				String pattern = params.get("pattern");
				String replacement = params.get("replacement");
				FrameBlock ret = target.replaceOperations(pattern, replacement);
				ec.setFrameOutput(output.getName(), ret);
				ec.releaseFrameInput(params.get("target"));
			} else{
				MatrixObject targetObj = ec.getMatrixObject(params.get("target"));
				MatrixBlock target = targetObj.acquireRead();
				double pattern = Double.parseDouble(params.get("pattern"));
				double replacement = Double.parseDouble(params.get("replacement"));
				MatrixBlock ret = target.replaceOperations(new MatrixBlock(), pattern, replacement, 
					InfrastructureAnalyzer.getLocalParallelism());
				if( ret == target ) //shallow copy (avoid bufferpool pollution)
					ec.setVariable(output.getName(), targetObj);
				else
					ec.setMatrixOutput(output.getName(), ret);
				targetObj.release();
			}
		}
		else if(opcode.equals(Opcodes.LOWERTRI.toString()) || opcode.equals(Opcodes.UPPERTRI.toString())) {
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			boolean lower = opcode.equals(Opcodes.LOWERTRI.toString());
			boolean diag = Boolean.parseBoolean(params.get("diag"));
			boolean values = Boolean.parseBoolean(params.get("values"));
			MatrixBlock ret = target.extractTriangular(new MatrixBlock(), lower, diag, values);
			ec.setMatrixOutput(output.getName(), ret);
			ec.releaseMatrixInput(params.get("target"));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.REXPAND.toString())) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get("target"));

			// compute the result
			double maxVal = Double.parseDouble(params.get("max"));
			boolean dirVal = params.get("dir").equals("rows");
			boolean cast = Boolean.parseBoolean(params.get("cast"));
			boolean ignore = Boolean.parseBoolean(params.get("ignore"));
			int numThreads = Integer.parseInt(params.get("k"));
			MatrixBlock ret = target.rexpandOperations(new MatrixBlock(), maxVal, dirVal, cast, ignore, numThreads);

			// release locks
			ec.setMatrixOutput(output.getName(), ret);
			ec.releaseMatrixInput(params.get("target"));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TOKENIZE.toString())) {
			// acquire locks
			FrameBlock data = ec.getFrameInput(params.get("target"));

			// compute tokenizer
			Tokenizer tokenizer = TokenizerFactory.createTokenizer(getParameterMap().get("spec"),
				Integer.parseInt(getParameterMap().get("max_tokens")));
			FrameBlock fbout = tokenizer.tokenize(data, OptimizerUtils.getTokenizeNumThreads());

			// release locks
			ec.setFrameOutput(output.getName(), fbout);
			ec.releaseFrameInput(params.get("target"));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TRANSFORMAPPLY.toString())) {
			// acquire locks
			FrameBlock data = ec.getFrameInput(params.get("target"));
			FrameBlock meta = ec.getFrameInput(params.get("meta"));
			MatrixBlock embeddings = params.get("embedding") != null ? ec.getMatrixInput(params.get("embedding")) : null;
			String[] colNames = data.getColumnNames();

			// compute transformapply
			MultiColumnEncoder encoder = EncoderFactory
				.createEncoder(params.get("spec"), colNames, data.getNumColumns(), meta, embeddings);
			MatrixBlock mbout = encoder.apply(data, OptimizerUtils.getTransformNumThreads());

			// release locks
			ec.setMatrixOutput(output.getName(), mbout);
			ec.releaseFrameInput(params.get("target"));
			ec.releaseFrameInput(params.get("meta"));
			if(params.get("embedding") != null)
				ec.releaseMatrixInput(params.get("embedding"));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TRANSFORMDECODE.toString())) {
			// acquire locks
			MatrixBlock data = ec.getMatrixInput(params.get("target"));
			FrameBlock meta = ec.getFrameInput(params.get("meta"));
			String[] colnames = meta.getColumnNames();


			ColumnDecoder decoder = ColumnDecoderFactory
					.createDecoder(getParameterMap().get("spec"), colnames, null, meta, meta.getNumColumns());
			FrameBlock out = new FrameBlock(decoder.getMultiSchema());
			FrameBlock fbout = decoder.columnDecode(data, out);
			fbout.setColumnNames(Arrays.copyOfRange(colnames, 0, fbout.getNumColumns()));


			//Decoder decoder = DecoderFactory
			//	.createDecoder(getParameterMap().get("spec"), colnames, null, meta, data.getNumColumns());
			//FrameBlock fbout = decoder.decode(data, new FrameBlock(decoder.getSchema()));
			//fbout.setColumnNames(Arrays.copyOfRange(colnames, 0, fbout.getNumColumns()));

			// release locks
			ec.setFrameOutput(output.getName(), fbout);
			ec.releaseMatrixInput(params.get("target"));
			ec.releaseFrameInput(params.get("meta"));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TRANSFORMCOLMAP.toString())) {
			// acquire locks
			FrameBlock meta = ec.getFrameInput(params.get("target"));
			String[] colNames = meta.getColumnNames();

			// compute transformapply
			MultiColumnEncoder encoder = EncoderFactory
				.createEncoder(params.get("spec"), colNames, meta.getNumColumns(), null, null);
			MatrixBlock mbout = encoder.getColMapping(meta);

			// release locks
			ec.setMatrixOutput(output.getName(), mbout);
			ec.releaseFrameInput(params.get("target"));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TRANSFORMMETA.toString())) {
			// get input spec and path
			String spec = getParameterMap().get("spec");
			String path = getParameterMap().get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_MTD);
			String delim = getParameterMap().getOrDefault("sep", TfUtils.TXMTD_SEP);

			// execute transform meta data read
			FrameBlock meta = null;
			try {
				meta = TfMetaUtils.readTransformMetaDataFromFile(spec, path, delim);
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}

			// release locks
			ec.setFrameOutput(output.getName(), meta);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TOSTRING.toString())) {
			// handle input parameters
			int rows = (getParam("rows") != null) ? Integer.parseInt(getParam("rows")) : TOSTRING_MAXROWS;
			int cols = (getParam("cols") != null) ? Integer.parseInt(getParam("cols")) : TOSTRING_MAXCOLS;
			int decimal = (getParam("decimal") != null) ? Integer.parseInt(getParam("decimal")) : TOSTRING_DECIMAL;
			boolean sparse = (getParam("sparse") != null) ? Boolean.parseBoolean(getParam("sparse")) : TOSTRING_SPARSE;
			String separator = (getParam("sep") != null) ? getParam("sep") : TOSTRING_SEPARATOR;
			String lineSeparator = (getParam("linesep") != null) ? getParam("linesep") : TOSTRING_LINESEPARATOR;

			// get input matrix/frame and convert to string
			String out = null;

			Data cacheData = ec.getVariable(getParam("target"));
			if(cacheData instanceof MatrixObject) {
				MatrixBlock matrix = ((MatrixObject) cacheData).acquireRead();
				warnOnTrunction(matrix, rows, cols);
				out = DataConverter.toString(matrix, sparse, separator, lineSeparator, rows, cols, decimal);
			}
			else if(cacheData instanceof TensorObject) {
				TensorBlock tensor = ((TensorObject) cacheData).acquireRead();
				// TODO improve truncation to check all dimensions
				warnOnTrunction(tensor, rows, cols);
				out = DataConverter.toString(tensor, sparse, separator, lineSeparator, "[", "]", rows, cols, decimal);
			}
			else if(cacheData instanceof FrameObject) {
				FrameBlock frame = ((FrameObject) cacheData).acquireRead();
				warnOnTrunction(frame, rows, cols);
				out = DataConverter.toString(frame, sparse, separator, lineSeparator, rows, cols, decimal);
			}
			else if(cacheData instanceof ListObject) {
				out = DataConverter.toString((ListObject) cacheData,
					rows, cols, sparse, separator, lineSeparator, rows, cols, decimal);
			}
			else {
				throw new DMLRuntimeException("toString only converts "
					+ "matrix, tensors, lists or frames to string: "+cacheData.getClass().getSimpleName());
			}
			if(!(cacheData instanceof ListObject)) {
				ec.releaseCacheableData(getParam("target"));
			}
			ec.setScalarOutput(output.getName(), new StringObject(out));
		}
		else if(opcode.equals(Opcodes.NVLIST.toString())) {
			// obtain all input data objects and names in insertion order
			List<Data> data = params.values().stream()
				.map(d -> ec.containsVariable(d) ? ec.getVariable(d) :
					ScalarObjectFactory.createScalarObject(d))
				.collect(Collectors.toList());
			List<String> names = new ArrayList<>(params.keySet());

			ListObject list = null;
			if (DMLScript.LINEAGE) {
				CPOperand[] listOperands = names.stream().map(n -> ec.containsVariable(params.get(n)) 
						? new CPOperand(n, ec.getVariable(params.get(n))) 
						: getStringLiteral(n)).toArray(CPOperand[]::new);
				LineageItem[] liList = LineageItemUtils.getLineage(ec, listOperands);
				// create list object over all inputs w/ the corresponding lineage items
				list = new ListObject(data, names, Arrays.asList(liList));
			}
			else
				// create list object over all inputs
				list = new ListObject(data, names);

			list.deriveAndSetStatusFromData();

			ec.setVariable(output.getName(), list);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode : " + opcode);
		}
	}

	private void warnOnTrunction(CacheBlock<?> data, int rows, int cols) {
		// warn on truncation because users might not be aware and use toString for verification
		if((getParam("rows") == null && data.getNumRows() > rows) ||
			(getParam("cols") == null && data.getNumColumns() > cols)) {
			LOG.warn("Truncating " + data.getClass().getSimpleName() + " of size " + data.getNumRows() + "x" + data
				.getNumColumns() + " to " + rows + "x" + cols + ". " + "Use toString(X, rows=..., cols=...) if necessary.");
		}
	}

	private void warnOnTrunction(TensorBlock data, int rows, int cols) {
		// warn on truncation because users might not be aware and use toString for verification
		if((getParam("rows") == null && data.getDim(0) > rows) || (getParam("cols") == null && data.getDim(1) > cols)) {
			StringBuilder sb = new StringBuilder();
			IntStream.range(0, data.getNumDims()).forEach((i) -> {
				if((i == data.getNumDims() - 1))
					sb.append(data.getDim(i));
				else
					sb.append(data.getDim(i)).append("x");
			});
			LOG.warn("Truncating " + data.getClass().getSimpleName() + " of size " + sb.toString() + " to " + rows + "x"
				+ cols + ". " + "Use toString(X, rows=..., cols=...) if necessary.");
		}
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		String opcode = getOpcode();
		if(opcode.equalsIgnoreCase(Opcodes.CONTAINS.toString())) {
			CPOperand target = getTargetOperand();
			CPOperand pattern = getFP64Literal("pattern");
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, pattern)));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.GROUPEDAGG.toString())) {
			CPOperand target = getTargetOperand();
			CPOperand groups = new CPOperand(params.get(Statement.GAGG_GROUPS), ValueType.FP64, DataType.MATRIX);
			String wt = params.containsKey(Statement.GAGG_WEIGHTS) ? params.get(Statement.GAGG_WEIGHTS) : String
				.valueOf(-1);
			CPOperand weights = new CPOperand(wt, ValueType.FP64, DataType.MATRIX);
			CPOperand fn = getStringLiteral(Statement.GAGG_FN);
			String ng = params.containsKey(Statement.GAGG_NUM_GROUPS) ? params.get(Statement.GAGG_NUM_GROUPS) : String
				.valueOf(-1);
			CPOperand ngroups = new CPOperand(ng, ValueType.INT64, DataType.SCALAR, true);
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, groups, weights, fn, ngroups)));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.RMEMPTY.toString())) {
			CPOperand target = getTargetOperand();
			CPOperand margin = getStringLiteral("margin");
			String sl = params.containsKey("select") ? params.get("select") : String.valueOf(-1);
			CPOperand select = new CPOperand(sl, ValueType.FP64, DataType.MATRIX);
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, margin, select)));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.REPLACE.toString())) {
			CPOperand target = getTargetOperand();
			CPOperand pattern = getFP64Literal("pattern");
			CPOperand replace = getFP64Literal("replacement");
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, pattern, replace)));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.REXPAND.toString())) {
			CPOperand target = getTargetOperand();
			CPOperand max = getFP64Literal("max");
			CPOperand dir = getStringLiteral("dir");
			CPOperand cast = getBoolLiteral("cast");
			CPOperand ignore = getBoolLiteral("ignore");
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, max, dir, cast, ignore)));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.LOWERTRI.toString()) || opcode.equalsIgnoreCase(Opcodes.UPPERTRI.toString())) {
			CPOperand target = getTargetOperand();
			CPOperand lower = getBoolLiteral(Opcodes.LOWERTRI.toString());
			CPOperand diag = getBoolLiteral("diag");
			CPOperand values = getBoolLiteral("values");
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, lower, diag, values)));
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TRANSFORMDECODE.toString()) || opcode.equalsIgnoreCase(Opcodes.TRANSFORMAPPLY.toString())) {
			CPOperand target = new CPOperand(params.get("target"), ValueType.FP64, DataType.FRAME);
			CPOperand meta = getLiteral("meta", ValueType.UNKNOWN, DataType.FRAME);
			CPOperand spec = new CPOperand(params.get("spec"), ValueType.STRING, DataType.SCALAR);
			return Pair.of(output.getName(),
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, meta, spec)));
		}
		else if (opcode.equalsIgnoreCase(Opcodes.NVLIST.toString()) || opcode.equalsIgnoreCase(Opcodes.AUTODIFF.toString())) {
			List<String> names = new ArrayList<>(params.keySet());
			CPOperand[] listOperands = names.stream().map(n -> ec.containsVariable(params.get(n)) 
					? new CPOperand(n, ec.getVariable(params.get(n))) 
					: getStringLiteral(n)).toArray(CPOperand[]::new);
			return Pair.of(output.getName(), 
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, listOperands)));
		}
		else {
			// NOTE: for now, we cannot have a generic fall through path, because the
			// data and value types of parmeters are not compiled into the instruction
			throw new DMLRuntimeException("Unsupported lineage tracing for: " + opcode);
		}
	}

	public CacheableData<?> getTarget(ExecutionContext ec) {
		return ec.getCacheableData(params.get("target"));
	}

	private CPOperand getTargetOperand() {
		return new CPOperand(params.get("target"), ValueType.FP64, DataType.MATRIX);
	}

	private CPOperand getFP64Literal(String name) {
		return getLiteral(name, ValueType.FP64);
	}

	private CPOperand getStringLiteral(String name) {
		return getLiteral(name, ValueType.STRING);
	}

	private CPOperand getBoolLiteral(String name) {
		return getLiteral(name, ValueType.BOOLEAN);
	}

	private CPOperand getLiteral(String name, ValueType vt) {
		return new CPOperand(params.get(name), vt, DataType.SCALAR, true);
	}

	private CPOperand getLiteral(String name, ValueType vt, DataType dt) {
		return new CPOperand(params.get(name), vt, dt);
	}
}
