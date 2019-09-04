/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.parser.ParameterizedBuiltinFunctionExpression;
import org.tugraz.sysds.parser.Statement;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheBlock;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheableData;
import org.tugraz.sysds.runtime.controlprogram.caching.FrameObject;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.caching.TensorObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.functionobjects.ParameterizedBuiltin;
import org.tugraz.sysds.runtime.functionobjects.ValueFunction;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.matrix.operators.SimpleOperator;
import org.tugraz.sysds.runtime.transform.TfUtils;
import org.tugraz.sysds.runtime.transform.decode.Decoder;
import org.tugraz.sysds.runtime.transform.decode.DecoderFactory;
import org.tugraz.sysds.runtime.transform.encode.Encoder;
import org.tugraz.sysds.runtime.transform.encode.EncoderFactory;
import org.tugraz.sysds.runtime.transform.meta.TfMetaUtils;
import org.tugraz.sysds.runtime.util.DataConverter;

public class ParameterizedBuiltinCPInstruction extends ComputationCPInstruction {
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
	
	public HashMap<String,String> getParameterMap() { 
		return params; 
	}
	
	public String getParam(String key) {
		return getParameterMap().get(key);
	}
	
	public static LinkedHashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		LinkedHashMap<String,String> paramMap = new LinkedHashMap<>();
		
		// all parameters are of form <name=value>
		String[] parts;
		for ( int i=1; i <= params.length-2; i++ ) {
			parts = params[i].split(Lop.NAME_VALUE_SEPARATOR);
			paramMap.put(parts[0], parts[1]);
		}
		
		return paramMap;
	}
	
	public static ParameterizedBuiltinCPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand( parts[parts.length-1] ); 

		// process remaining parts and build a hash map
		LinkedHashMap<String,String> paramsMap = constructParameterMap(parts);

		// determine the appropriate value function
		ValueFunction func = null;
		if ( opcode.equalsIgnoreCase("cdf") ) {
			if ( paramsMap.get("dist") == null ) 
				throw new DMLRuntimeException("Invalid distribution: " + str);
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode, paramsMap.get("dist"));
			// Determine appropriate Function Object based on opcode
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("invcdf") ) {
			if ( paramsMap.get("dist") == null ) 
				throw new DMLRuntimeException("Invalid distribution: " + str);
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode, paramsMap.get("dist"));
			// Determine appropriate Function Object based on opcode
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("groupedagg")) {
			// check for mandatory arguments
			String fnStr = paramsMap.get("fn");
			if ( fnStr == null ) 
				throw new DMLRuntimeException("Function parameter is missing in groupedAggregate.");
			if ( fnStr.equalsIgnoreCase("centralmoment") ) {
				if ( paramsMap.get("order") == null )
					throw new DMLRuntimeException("Mandatory \"order\" must be specified when fn=\"centralmoment\" in groupedAggregate.");
			}
			
			Operator op = InstructionUtils.parseGroupedAggOperator(fnStr, paramsMap.get("order"));
			return new ParameterizedBuiltinCPInstruction(op, paramsMap, out, opcode, str);
		} else if (opcode.equalsIgnoreCase("rmempty")
				|| opcode.equalsIgnoreCase("replace")
				|| opcode.equalsIgnoreCase("rexpand")
				|| opcode.equalsIgnoreCase("lowertri")
				|| opcode.equalsIgnoreCase("uppertri")) {
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		} else if (opcode.equals("transformapply")
				|| opcode.equals("transformdecode")
				|| opcode.equals("transformcolmap")
				|| opcode.equals("transformmeta")
				|| opcode.equals("toString")
				|| opcode.equals("nvlist")) {
			return new ParameterizedBuiltinCPInstruction(null, paramsMap, out, opcode, str);
		} else if ("paramserv".equals(opcode)) {
			return new ParamservBuiltinCPInstruction(null, paramsMap, out, opcode, str);
		} else {
			throw new DMLRuntimeException("Unknown opcode (" + opcode + ") for ParameterizedBuiltin Instruction.");
		}

	}

	@Override 
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		ScalarObject sores = null;
		if ( opcode.equalsIgnoreCase("cdf")) {
			SimpleOperator op = (SimpleOperator) _optr;
			double result =  op.fn.execute(params);
			sores = new DoubleObject(result);
			ec.setScalarOutput(output.getName(), sores);
		} 
		else if ( opcode.equalsIgnoreCase("invcdf")) {
			SimpleOperator op = (SimpleOperator) _optr;
			double result =  op.fn.execute(params);
			sores = new DoubleObject(result);
			ec.setScalarOutput(output.getName(), sores);
		} 
		else if ( opcode.equalsIgnoreCase("groupedagg") ) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get(Statement.GAGG_TARGET));
			MatrixBlock groups = ec.getMatrixInput(params.get(Statement.GAGG_GROUPS));
			MatrixBlock weights= null;
			if ( params.get(Statement.GAGG_WEIGHTS) != null )
				weights = ec.getMatrixInput(params.get(Statement.GAGG_WEIGHTS));
			
			int ngroups = -1;
			if ( params.get(Statement.GAGG_NUM_GROUPS) != null) {
				ngroups = (int) Double.parseDouble(params.get(Statement.GAGG_NUM_GROUPS));
			}
			
			// compute the result
			int k = Integer.parseInt(params.get("k")); //num threads
			MatrixBlock soresBlock = groups.groupedAggOperations(target, weights, new MatrixBlock(), ngroups, _optr, k);
			
			ec.setMatrixOutput(output.getName(), soresBlock);
			// release locks
			target = groups = weights = null;
			ec.releaseMatrixInput(params.get(Statement.GAGG_TARGET));
			ec.releaseMatrixInput(params.get(Statement.GAGG_GROUPS));
			if ( params.get(Statement.GAGG_WEIGHTS) != null )
				ec.releaseMatrixInput(params.get(Statement.GAGG_WEIGHTS));
			
		}
		else if ( opcode.equalsIgnoreCase("rmempty") ) {
			String margin = params.get("margin");
			if( !(margin.equals("rows") || margin.equals("cols")) )
				throw new DMLRuntimeException("Unspupported margin identifier '"+margin+"'.");
			
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			MatrixBlock select = params.containsKey("select") ?
				ec.getMatrixInput(params.get("select")):null;
			
			// compute the result
			boolean emptyReturn = Boolean.parseBoolean(params.get("empty.return").toLowerCase());
			MatrixBlock soresBlock = target.removeEmptyOperations(new MatrixBlock(),
				margin.equals("rows"), emptyReturn, select);
			
			//release locks
			ec.setMatrixOutput(output.getName(), soresBlock);
			ec.releaseMatrixInput(params.get("target"));
			if (params.containsKey("select"))
				ec.releaseMatrixInput(params.get("select"));
		}
		else if ( opcode.equalsIgnoreCase("replace") ) {
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			double pattern = Double.parseDouble( params.get("pattern") );
			double replacement = Double.parseDouble( params.get("replacement") );
			MatrixBlock ret = (MatrixBlock) target.replaceOperations(new MatrixBlock(), pattern, replacement);
			ec.setMatrixOutput(output.getName(), ret);
			ec.releaseMatrixInput(params.get("target"));
		}
		else if ( opcode.equals("lowertri") || opcode.equals("uppertri")) {
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			boolean lower = opcode.equals("lowertri");
			boolean diag = Boolean.parseBoolean(params.get("diag"));
			boolean values = Boolean.parseBoolean(params.get("values"));
			MatrixBlock ret = (MatrixBlock) target.extractTriangular(new MatrixBlock(), lower, diag, values);
			ec.setMatrixOutput(output.getName(), ret);
			ec.releaseMatrixInput(params.get("target"));
		}
		else if ( opcode.equalsIgnoreCase("rexpand") ) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			
			// compute the result
			double maxVal = Double.parseDouble( params.get("max") );
			boolean dirVal = params.get("dir").equals("rows");
			boolean cast = Boolean.parseBoolean(params.get("cast"));
			boolean ignore = Boolean.parseBoolean(params.get("ignore"));
			int numThreads = Integer.parseInt(params.get("k"));
			MatrixBlock ret = (MatrixBlock) target.rexpandOperations(
				new MatrixBlock(), maxVal, dirVal, cast, ignore, numThreads);
			
			//release locks
			ec.setMatrixOutput(output.getName(), ret);
			ec.releaseMatrixInput(params.get("target"));
		}
		else if ( opcode.equalsIgnoreCase("transformapply")) {
			//acquire locks
			FrameBlock data = ec.getFrameInput(params.get("target"));
			FrameBlock meta = ec.getFrameInput(params.get("meta"));
			String[] colNames = data.getColumnNames();
			
			//compute transformapply
			Encoder encoder = EncoderFactory.createEncoder(params.get("spec"), colNames, data.getNumColumns(), meta);
			MatrixBlock mbout = encoder.apply(data, new MatrixBlock(data.getNumRows(), data.getNumColumns(), false));
			
			//release locks
			ec.setMatrixOutput(output.getName(), mbout);
			ec.releaseFrameInput(params.get("target"));
			ec.releaseFrameInput(params.get("meta"));
		}
		else if ( opcode.equalsIgnoreCase("transformdecode")) {
			//acquire locks
			MatrixBlock data = ec.getMatrixInput(params.get("target"));
			FrameBlock meta = ec.getFrameInput(params.get("meta"));
			String[] colnames = meta.getColumnNames();
			
			//compute transformdecode
			Decoder decoder = DecoderFactory.createDecoder(
				getParameterMap().get("spec"), colnames, null, meta, data.getNumColumns());
			FrameBlock fbout = decoder.decode(data, new FrameBlock(decoder.getSchema()));
			fbout.setColumnNames(Arrays.copyOfRange(colnames, 0, fbout.getNumColumns()));
			
			//release locks
			ec.setFrameOutput(output.getName(), fbout);
			ec.releaseMatrixInput(params.get("target"));
			ec.releaseFrameInput(params.get("meta"));
		}
		else if ( opcode.equalsIgnoreCase("transformcolmap")) {
			//acquire locks
			FrameBlock meta = ec.getFrameInput(params.get("target"));
			String[] colNames = meta.getColumnNames();
			
			//compute transformapply
			Encoder encoder = EncoderFactory.createEncoder(params.get("spec"), colNames, meta.getNumColumns(), null);
			MatrixBlock mbout = encoder.getColMapping(meta, new MatrixBlock(meta.getNumColumns(), 3, false));
			
			//release locks
			ec.setMatrixOutput(output.getName(), mbout);
			ec.releaseFrameInput(params.get("target"));
		}
		else if ( opcode.equalsIgnoreCase("transformmeta")) {
			//get input spec and path
			String spec = getParameterMap().get("spec");
			String path = getParameterMap().get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_MTD);
			String delim = getParameterMap().containsKey("sep") ? getParameterMap().get("sep") : TfUtils.TXMTD_SEP;
			
			//execute transform meta data read
			FrameBlock meta = null;
			try {
				meta = TfMetaUtils.readTransformMetaDataFromFile(spec, path, delim);
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
			
			//release locks
			ec.setFrameOutput(output.getName(), meta);
		}
		else if ( opcode.equalsIgnoreCase("toString")) {
			//handle input parameters
			int rows = (getParam("rows")!=null) ? Integer.parseInt(getParam("rows")) : TOSTRING_MAXROWS;
			int cols = (getParam("cols") != null) ? Integer.parseInt(getParam("cols")) : TOSTRING_MAXCOLS;
			int decimal = (getParam("decimal") != null) ? Integer.parseInt(getParam("decimal")) : TOSTRING_DECIMAL;
			boolean sparse = (getParam("sparse") != null) ? Boolean.parseBoolean(getParam("sparse")) : TOSTRING_SPARSE;
			String separator = (getParam("sep") != null) ? getParam("sep") : TOSTRING_SEPARATOR;
			String lineseparator = (getParam("linesep") != null) ? getParam("linesep") : TOSTRING_LINESEPARATOR;
			
			//get input matrix/frame and convert to string
			String out = null;
			CacheableData<?> data = ec.getCacheableData(getParam("target"));
			if( data instanceof MatrixObject ) {
				MatrixBlock matrix = (MatrixBlock) data.acquireRead();
				warnOnTrunction(matrix, rows, cols);
				out = DataConverter.toString(matrix, sparse, separator, lineseparator, rows, cols, decimal);
			}
			else if( data instanceof TensorObject ) {
				TensorBlock tensor = (TensorBlock) data.acquireRead();
				// TODO improve truncation to check all dimensions
				warnOnTrunction(tensor, rows, cols);
				out = DataConverter.toString(tensor, sparse, separator,
					lineseparator, "[", "]", rows, cols, decimal);
			}
			else if( data instanceof FrameObject ) {
				FrameBlock frame = (FrameBlock) data.acquireRead();
				warnOnTrunction(frame, rows, cols);
				out = DataConverter.toString(frame, sparse, separator, lineseparator, rows, cols, decimal);
			}
			else {
				throw new DMLRuntimeException("toString only converts matrix, tensors or frames to string");
			}
			ec.releaseCacheableData(getParam("target"));
			ec.setScalarOutput(output.getName(), new StringObject(out));
		}
		else if( opcode.equals("nvlist") ) {
			//obtain all input data objects and names in insertion order
			List<Data> data = params.values().stream().map(d -> ec.containsVariable(d) ?
				ec.getVariable(d) : new StringObject(d)).collect(Collectors.toList());
			List<String> names = new ArrayList<>(params.keySet());
			
			//create list object over all inputs
			ListObject list = new ListObject(data, names);
			list.deriveAndSetStatusFromData();
			
			ec.setVariable(output.getName(), list);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode : " + opcode);
		}
	}
	
	private void warnOnTrunction(CacheBlock data, int rows, int cols) {
		//warn on truncation because users might not be aware and use toString for verification
		if( (getParam("rows")==null && data.getNumRows()>rows)
			|| (getParam("cols")==null && data.getNumColumns()>cols) )
		{
			LOG.warn("Truncating "+data.getClass().getSimpleName()+" of size "
				+ data.getNumRows()+"x"+data.getNumColumns()+" to "+rows+"x"+cols+". "
				+ "Use toString(X, rows=..., cols=...) if necessary.");
		}
	}

	private void warnOnTrunction(TensorBlock data, int rows, int cols) {
		//warn on truncation because users might not be aware and use toString for verification
		if( (getParam("rows")==null && data.getDim(0)>rows)
			|| (getParam("cols")==null && data.getDim(1)>cols) )
		{
			StringBuilder sb = new StringBuilder();
			IntStream.range(0, data.getNumDims()).forEach((i) -> {
				if ((i == data.getNumDims() - 1))
					sb.append(data.getDim(i));
				else
					sb.append(data.getDim(i)).append("x");
			});
			LOG.warn("Truncating "+data.getClass().getSimpleName()+" of size "+sb.toString()+" to "+rows+"x"+cols+". "
					+ "Use toString(X, rows=..., cols=...) if necessary.");
		}
	}
}
