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

package org.apache.sysds.hops.rewrite;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpDnn;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.DnnOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LeftIndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.MemoTable;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.instructions.cp.StringInitCPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;

public class HopRewriteUtils {
	private static final Log LOG = LogFactory.getLog(HopRewriteUtils.class.getName());

	public static boolean isValueTypeCast( OpOp1 op ) {
		return op == OpOp1.CAST_AS_BOOLEAN
			|| op == OpOp1.CAST_AS_INT
			|| op == OpOp1.CAST_AS_DOUBLE;
	}
	
	//////////////////////////////////
	// literal handling

	public static boolean getBooleanValue( LiteralOp op ) {
		switch( op.getValueType() ) {
			case FP64:    return op.getDoubleValue() != 0; 
			case INT64:   return op.getLongValue()   != 0;
			case BOOLEAN: return op.getBooleanValue();
			default: throw new HopsException("Invalid boolean value: "+op.getValueType());
		}
	}

	public static boolean getBooleanValueSafe( LiteralOp op ) {
		try {
			switch( op.getValueType() ) {
				case FP64:    return op.getDoubleValue() != 0; 
				case INT64:   return op.getLongValue()   != 0;
				case BOOLEAN: return op.getBooleanValue();
				default: throw new HopsException("Invalid boolean value: "+op.getValueType());
			}
		}
		catch(Exception ex){
			//silently ignore error
		}
		
		return false;
	}

	public static double getDoubleValue( LiteralOp op ) {
		switch( op.getValueType() ) {
			case STRING:
			case FP64:    return op.getDoubleValue(); 
			case INT64:   return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			default: throw new HopsException("Invalid double value: "+op.getValueType());
		}
	}
	
	public static double getDoubleValueSafe( LiteralOp op ) {
		switch( op.getValueType() ) {
			case FP64:  return op.getDoubleValue(); 
			case INT64:     return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			default: return Double.MAX_VALUE;
		}
	}
	
	/**
	 * Return the int value of a LiteralOp (as a long).
	 *
	 * Note: For comparisons, this is *only* to be used in situations
	 * in which the value is absolutely guaranteed to be an integer.
	 * Otherwise, a safer alternative is `getDoubleValue`.
	 * 
	 * @param op literal operator
	 * @return long value of literal op
	 */
	public static long getIntValue( LiteralOp op ) {
		switch( op.getValueType() ) {
			case FP64:  return UtilFunctions.toLong(op.getDoubleValue());
			case STRING:
			case INT64:     return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			default: throw new HopsException("Invalid int value: "+op.getValueType());
		}
	}
	
	public static long getIntValueSafe( Hop op ) {
		return getIntValueSafe((LiteralOp) op);
	}
	
	public static long getIntValueSafe( LiteralOp op ) {
		switch( op.getValueType() ) {
			case FP64:    return UtilFunctions.toLong(op.getDoubleValue());
			case INT64:   return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			default: return Long.MAX_VALUE;
		}
	}
	
	public static boolean isLiteralOfValue( Hop hop, Double... val ) {
		return Arrays.stream(val).anyMatch(d -> isLiteralOfValue(hop, d));
	}
	
	public static boolean isLiteralOfValue( Hop hop, double val ) {
		return (hop instanceof LiteralOp 
			&& (hop.getValueType()==ValueType.FP64 || hop.getValueType()==ValueType.INT64)
			&& getDoubleValueSafe((LiteralOp)hop)==val);
	}
	
	public static boolean isLiteralOfValue(Hop hop, String val) {
		return hop instanceof LiteralOp 
			&& ((LiteralOp)hop).getStringValue().equals(val);
	}
	
	public static boolean isLiteralOfValue( Hop hop, boolean val ) {
		try {
			return (hop instanceof LiteralOp 
				&& (hop.getValueType()==ValueType.BOOLEAN)
				&& ((LiteralOp)hop).getBooleanValue()==val);
		}
		catch(HopsException ex) {
			throw new RuntimeException(ex);
		}
	}
	
	public static ScalarObject getScalarObject( LiteralOp op )
	{
		try {
			return ScalarObjectFactory
				.createScalarObject(op.getValueType(), op);
		}
		catch(Exception ex) {
			throw new RuntimeException("Failed to create scalar object for constant. Continue.", ex);
		}
	}
	

	///////////////////////////////////
	// hop dag transformations
	
	

	public static int getChildReferencePos( Hop parent, Hop child ) {
		return parent.getInput().indexOf(child);
	}
	
	public static void removeChildReference( Hop parent, Hop child ) {
		parent.getInput().remove( child );
		child.getParent().remove( parent );
	}
	
	public static void removeChildReferenceByPos( Hop parent, Hop child, int posChild ) {
		parent.getInput().remove( posChild );
		child.getParent().remove( parent );
	}

	public static void removeAllChildReferences( Hop parent )
	{
		//remove parent reference from all childs
		for( Hop child : parent.getInput() )
			child.getParent().remove(parent);
		
		//remove all child references
		parent.getInput().clear();
	}
	
	public static void addChildReference( Hop parent, Hop child ) {
		parent.getInput().add( child );
		child.getParent().add( parent );
	}
	
	public static void addChildReference( Hop parent, Hop child, int pos ){
		parent.getInput().add( pos, child );
		child.getParent().add( parent );
	}

	/**
	 * Replace an old Hop with a replacement Hop.
	 * If the old Hop has no parents, then return the replacement.
	 * Otherwise rewire each of the Hop's parents into the replacement and return the replacement.
	 * @param hold To be replaced
	 * @param hnew The replacement
	 * @return hnew
	 */
	public static Hop rewireAllParentChildReferences( Hop hold, Hop hnew ) {
		List<Hop> parents = hold.getParent();
		while (!parents.isEmpty())
			HopRewriteUtils.replaceChildReference(parents.get(0), hold, hnew);
		return hnew;
	}
	
	public static void replaceChildReference( Hop parent, Hop inOld, Hop inNew ) {
		int pos = getChildReferencePos(parent, inOld);
		removeChildReferenceByPos(parent, inOld, pos);
		addChildReference(parent, inNew, pos);
		parent.refreshSizeInformation();
	}
	
	public static void replaceChildReference( Hop parent, Hop inOld, Hop inNew, int pos ) {
		replaceChildReference(parent, inOld, inNew, pos, true);
	}
	
	public static void replaceChildReference( Hop parent, Hop inOld, Hop inNew, int pos, boolean refresh ) {
		removeChildReferenceByPos(parent, inOld, pos);
		addChildReference(parent, inNew, pos);
		if( refresh )
			parent.refreshSizeInformation();
	}
	
	public static void cleanupUnreferenced( Hop... inputs ) {
		for( Hop input : inputs )
			if( input.getParent().isEmpty() )
				removeAllChildReferences(input);
	}
	
	public static Hop getOtherInput(Hop hop, Hop input) {
		for( Hop c : hop.getInput() )
			if( c != input )
				return c;
		return null;
	}
	
	public static Hop getLargestInput(Hop hop) {
		return hop.getInput().stream()
			.max(Comparator.comparing(h -> h.getLength())).orElse(null);
	}
	
	public static Hop createDataGenOp( Hop input, double value ) 
	{
		Hop rows = input.rowsKnown() ? new LiteralOp(input.getDim1()) : 
			new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT64, OpOp1.NROW, input);
		Hop cols = input.colsKnown() ? new LiteralOp(input.getDim2()) :
			new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT64, OpOp1.NCOL, input);
		Hop val = new LiteralOp(value);
		
		HashMap<String, Hop> params = new HashMap<>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_LAMBDA, new LiteralOp(-1.0));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp(1.0));
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(OpOpDG.RAND, new DataIdentifier("tmp"), params);
		datagen.setBlocksize(input.getBlocksize());
		copyLineNumbers(input, datagen);
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	/**
	 * Assumes that min and max are literal ops, needs to be checked from outside.
	 * 
	 * @param inputGen input data gen op
	 * @param scale the scale
	 * @param shift the shift
	 * @return data gen op
	 */
	public static DataGenOp copyDataGenOp( DataGenOp inputGen, double scale, double shift ) 
	{
		HashMap<String, Integer> params = inputGen.getParamIndexMap();
		Hop min = inputGen.getInput().get(params.get(DataExpression.RAND_MIN));
		Hop max = inputGen.getInput().get(params.get(DataExpression.RAND_MAX));
		Hop pdf = inputGen.getInput().get(params.get(DataExpression.RAND_PDF));
		Hop mean = inputGen.getInput().get(params.get(DataExpression.RAND_LAMBDA));
		Hop sparsity = inputGen.getInput().get(params.get(DataExpression.RAND_SPARSITY));
		Hop seed = inputGen.getInput().get(params.get(DataExpression.RAND_SEED));
		
		//check for literal ops
		if( !(min instanceof LiteralOp) || !(max instanceof LiteralOp))
			return null;
		
		//scale and shift
		double smin = getDoubleValue((LiteralOp) min);
		double smax = getDoubleValue((LiteralOp) max);
		smin = smin * scale + shift;
		smax = smax * scale + shift;
		
		Hop sminHop = new LiteralOp(smin);
		Hop smaxHop = new LiteralOp(smax);
		
		HashMap<String, Hop> params2 = new HashMap<>();
		if( !params.containsKey(DataExpression.RAND_DIMS) ) {
			Hop rows = inputGen.getInput().get(params.get(DataExpression.RAND_ROWS));
			Hop cols = inputGen.getInput().get(params.get(DataExpression.RAND_COLS));
			params2.put(DataExpression.RAND_ROWS, rows);
			params2.put(DataExpression.RAND_COLS, cols);
		}
		else {
			Hop dims = inputGen.getInput().get(params.get(DataExpression.RAND_DIMS));
			params2.put(DataExpression.RAND_DIMS, dims);
		}
		params2.put(DataExpression.RAND_MIN, sminHop);
		params2.put(DataExpression.RAND_MAX, smaxHop);
		params2.put(DataExpression.RAND_PDF, pdf);
		params2.put(DataExpression.RAND_LAMBDA, mean);
		params2.put(DataExpression.RAND_SPARSITY, sparsity);
		params2.put(DataExpression.RAND_SEED, seed );
		
		//note internal refresh size information
		DataGenOp datagen = new DataGenOp(OpOpDG.RAND, new DataIdentifier("tmp"), params2);
		datagen.setBlocksize(inputGen.getBlocksize());
		copyLineNumbers(inputGen, datagen);
		
		if( smin==0 && smax==0 )
			datagen.setNnz(0);
			
		return datagen;
	}

	public static Hop createDataGenOpFromDims( Hop rows, Hop cols, double value ) {
		Hop val = new LiteralOp(value);

		HashMap<String, Hop> params = new HashMap<>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_LAMBDA, new LiteralOp(-1.0));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp(1.0));
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );

		//note internal refresh size information
		Hop datagen = new DataGenOp(OpOpDG.RAND, new DataIdentifier("tmp"), params);
		datagen.setBlocksize(1000);
		//copyLineNumbers(rowInput, datagen);

		if( value==0 )
			datagen.setNnz(0);

		return datagen;
	}
	
	public static Hop createDataGenOp( Hop rowInput, Hop colInput, double value ) 
	{
		Hop rows = rowInput.rowsKnown() ? new LiteralOp(rowInput.getDim1()) : 
			new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT64, OpOp1.NROW, rowInput);
		Hop cols = colInput.colsKnown() ? new LiteralOp(colInput.getDim2()) :
			new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT64, OpOp1.NCOL, colInput);
		Hop val = new LiteralOp(value);
		
		HashMap<String, Hop> params = new HashMap<>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_LAMBDA, new LiteralOp(-1.0));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp(1.0));
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(OpOpDG.RAND, new DataIdentifier("tmp"), params);
		datagen.setBlocksize(rowInput.getBlocksize());
		copyLineNumbers(rowInput, datagen);
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOp( Hop rowInput, boolean tRowInput, Hop colInput, boolean tColInput, double value ) 
	{
		long nrow = tRowInput ? rowInput.getDim2() : rowInput.getDim1();
		long ncol = tColInput ? colInput.getDim1() : rowInput.getDim2();
		
		Hop rows = (nrow>=0) ? new LiteralOp(nrow) : 
			new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT64, tRowInput?OpOp1.NCOL:OpOp1.NROW, rowInput);
		Hop cols = (ncol>=0) ? new LiteralOp(ncol) :
			new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT64, tColInput?OpOp1.NROW:OpOp1.NCOL, colInput);
		Hop val = new LiteralOp(value);
		
		HashMap<String, Hop> params = new HashMap<>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_LAMBDA,new LiteralOp(-1.0));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp(1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(OpOpDG.RAND, new DataIdentifier("tmp"), params);
		datagen.setBlocksize(colInput.getBlocksize());
		copyLineNumbers(rowInput, datagen);
		
		if( value==0 )
			datagen.setNnz(0);
		
		return datagen;
	}
	
	public static Hop createDataGenOpByVal(Hop rowInput, Hop colInput, Hop dimsInput, DataType dt, ValueType vt, double value) {
		Hop val = new LiteralOp(value);
		
		HashMap<String, Hop> params = new HashMap<>();
		if (dt.isTensor())
			params.put(DataExpression.RAND_DIMS, dimsInput);
		else {
			params.put(DataExpression.RAND_ROWS, rowInput);
			params.put(DataExpression.RAND_COLS, colInput);
		}
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_LAMBDA, new LiteralOp(-1.0));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp(1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		DataIdentifier di = new DataIdentifier("tmp");
		di.setDataType(dt);
		di.setValueType(vt);
		Hop datagen = new DataGenOp(OpOpDG.RAND, di, params);
		datagen.setBlocksize(rowInput.getBlocksize());
		copyLineNumbers(rowInput, datagen);
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOpByVal( List<LiteralOp> values, long rows, long cols ) 
	{
		StringBuilder sb = new StringBuilder();
		for(LiteralOp lit : values) {
			if(sb.length()>0)
				sb.append(StringInitCPInstruction.DELIM);
			sb.append(lit.getStringValue());
		}
		LiteralOp str = new LiteralOp(sb.toString());
		
		HashMap<String, Hop> params = new HashMap<>();
		params.put(DataExpression.RAND_ROWS, new LiteralOp(rows));
		params.put(DataExpression.RAND_COLS, new LiteralOp(cols));
		params.put(DataExpression.RAND_MIN, str);
		params.put(DataExpression.RAND_MAX, str);
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED));
		
		Hop datagen = new DataGenOp(OpOpDG.SINIT,
			new DataIdentifier("tmp", DataType.MATRIX), params);
		datagen.setBlocksize(ConfigurationManager.getBlocksize());
		copyLineNumbers(values.get(0), datagen);
		
		return datagen;
	}
	
	public static boolean isDataGenOp(Hop hop, OpOpDG... ops) {
		return (hop instanceof DataGenOp 
			&& ArrayUtils.contains(ops, ((DataGenOp)hop).getOp()));
	}
	
	public static boolean isDataGenOpWithLiteralInputs(Hop hop, OpOpDG... ops) {
		boolean ret = isDataGenOp(hop, ops);
		for( Hop c : hop.getInput() )
			ret &= c instanceof LiteralOp;
		return ret;
	}
	
	public static boolean isDataGenOpWithConstantValue(Hop hop) {
		return hop instanceof DataGenOp
			&& ((DataGenOp)hop).getOp()==OpOpDG.RAND
			&& ((DataGenOp)hop).hasConstantValue();
	}
	
	public static boolean isDataGenOpWithConstantValue(Hop hop, double value) {
		return hop instanceof DataGenOp
			&& ((DataGenOp)hop).getOp()==OpOpDG.RAND
			&& ((DataGenOp)hop).hasConstantValue(value);
	}
	
	public static boolean isDataGenOpWithNonDeterminism(Hop hop) {
		if (!isDataGenOp(hop, OpOpDG.RAND, OpOpDG.SAMPLE))
			return false;
		return isDataGenOp(hop, OpOpDG.SAMPLE) || (isDataGenOp(hop, OpOpDG.RAND) 
			&& !((DataGenOp)hop).hasConstantValue() && ((DataGenOp)hop).hasUnspecifiedSeed());
	}
	
	public static Hop getDataGenOpConstantValue(Hop hop) {
		return ((DataGenOp) hop).getConstantValue();
	}
	
	public static DataOp createTransientRead(String name, Hop h) {
		//note: different constructor necessary for formattype
		DataOp tread = new DataOp(name, h.getDataType(), h.getValueType(),
			OpOpData.TRANSIENTREAD, null, h.getDim1(), h.getDim2(), h.getNnz(),
			h.getUpdateType(), h.getBlocksize());
		tread.setVisited();
		copyLineNumbers(h, tread);
		return tread;
	}
	
	public static DataOp createTransientRead(String name, MatrixBlock mb) {
		DataOp tread = new DataOp(name, DataType.MATRIX, ValueType.FP64, OpOpData.TRANSIENTREAD,
			null, mb.getNumRows(), mb.getNumColumns(), mb.getNonZeros(), UpdateType.COPY, 
			ConfigurationManager.getBlocksize());
		tread.setVisited();
		copyLineNumbers(mb, tread);
		tread.setFileName(name);
		return tread;
	}
	
	public static DataOp createTransientRead(String name, MatrixObject mo) {
		DataOp tread = new DataOp(name, DataType.MATRIX, ValueType.FP64, OpOpData.TRANSIENTREAD,
			null, mo.getNumRows(), mo.getNumColumns(), mo.getNnz(), UpdateType.COPY, mo.getBlocksize());
		tread.setVisited();
		copyLineNumbers(mo, tread);
		tread.setFileName(name);
		return tread;
	}
	
	public static DataOp createTransientWrite(String name, Hop in) {
		return createDataOp(name, in, OpOpData.TRANSIENTWRITE);
	}
	
	public static DataOp createDataOp(String name, Hop in, OpOpData type) {
		DataOp dop = new DataOp(name, in.getDataType(),
			in.getValueType(), in, type, null);
		dop.setVisited();
		dop.setOutputParams(in.getDim1(), in.getDim2(), in.getNnz(),
			in.getUpdateType(), in.getBlocksize());
		copyLineNumbers(in, dop);
		return dop;
	}
	
	public static ReorgOp createTranspose(Hop input) {
		return createReorg(input, ReOrgOp.TRANS);
	}
	
	public static ReorgOp createReorg(Hop input, String rop) {
		return createReorg(input, ReOrgOp.valueOfByOpcode(rop));
	}
	
	public static ReorgOp createReorg(Hop input, ReOrgOp rop) {
		ReorgOp reorg = new ReorgOp(input.getName(), input.getDataType(), input.getValueType(), rop, input);
		reorg.setBlocksize(input.getBlocksize());
		copyLineNumbers(input, reorg);
		reorg.refreshSizeInformation();
		return reorg;
	}
	
	public static ReorgOp createReorg(List<Hop> inputs, ReOrgOp rop) {
		Hop main = inputs.get(0);
		ReorgOp reorg = new ReorgOp(main.getName(), main.getDataType(), main.getValueType(), rop, inputs);
		reorg.setBlocksize(main.getBlocksize());
		copyLineNumbers(main, reorg);
		reorg.refreshSizeInformation();
		return reorg;
	}
	
	public static UnaryOp createUnary(Hop input, String type) {
		return createUnary(input, OpOp1.valueOfByOpcode(type));
	}
	
	public static UnaryOp createUnary(Hop input, OpOp1 type)  {
		DataType dt = type.isScalarOutput() ? DataType.SCALAR :
			(type==OpOp1.CAST_AS_MATRIX) ? DataType.MATRIX : input.getDataType();
		ValueType vt = (type==OpOp1.CAST_AS_MATRIX) ? ValueType.FP64 : input.getValueType();
		UnaryOp unary = new UnaryOp(input.getName(), dt, vt, type, input);
		unary.setBlocksize(input.getBlocksize());
		if( type.isScalarOutput() || type == OpOp1.CAST_AS_MATRIX ) {
			int dim = type.isScalarOutput() ? 0 : 1;
			int blksz = (type==OpOp1.CAST_AS_SCALAR) ? 0 : ConfigurationManager.getBlocksize();
			setOutputParameters(unary, dim, dim, blksz, -1);
		}
		copyLineNumbers(input, unary);
		unary.refreshSizeInformation();
		return unary;
	}
	
	public static BinaryOp createBinaryMinus(Hop input) {
		return createBinary(new LiteralOp(0), input, OpOp2.MINUS);
	}
	
	public static BinaryOp createBinary(Hop input1, Hop input2, String op) {
		return createBinary(input1, input2, OpOp2.valueOfByOpcode(op), false);
	}
	
	public static BinaryOp createBinary(Hop input1, Hop input2, OpOp2 op) {
		return createBinary(input1, input2, op, false);
	}
	
	public static BinaryOp createBinary(Hop input1, Hop input2, OpOp2 op, boolean outer) {
		Hop mainInput = input1.getDataType().isMatrix() ? input1 :
			input2.getDataType().isMatrix() ? input2 : input1;
		BinaryOp bop = new BinaryOp(mainInput.getName(), mainInput.getDataType(),
			mainInput.getValueType(), op, input1, input2);
		//cleanup value type for relational operations
		if( bop.isPPredOperation() && bop.getDataType().isScalar() )
			bop.setValueType(ValueType.BOOLEAN);
		bop.setOuterVectorOperation(outer);
		bop.setBlocksize(mainInput.getBlocksize());
		copyLineNumbers(mainInput, bop);
		bop.refreshSizeInformation();
		return bop;
	}

	// To fix issues with createBinary, which does not always correctly set value types (e.g. INT-MATRIX+FLOAT-SCALAR -> bop(+)::INT)
	public static BinaryOp createAutoGeneratedBinary(Hop input1, Hop input2, OpOp2 op) {
		Hop mainInput = input1.getDataType().isMatrix() ? input1 :
				input2.getDataType().isMatrix() ? input2 : input1;
		BinaryOp bop = new BinaryOp(mainInput.getName(), getImplicitDataType(input1, input2),
				getImplicitValueType(input1, input2), op, input1, input2);
		//cleanup value type for relational operations
		if( bop.isPPredOperation() && bop.getDataType().isScalar() )
			bop.setValueType(ValueType.BOOLEAN);
		bop.setOuterVectorOperation(false);
		bop.setBlocksize(mainInput.getBlocksize());
		copyLineNumbers(mainInput, bop);
		bop.refreshSizeInformation();
		return bop;
	}

	public static DataType getImplicitDataType(Hop... inputs) {
		for (int i = 0; i < inputs.length; i++)
			if (inputs[i].getDataType().isMatrix())
				return inputs[i].getDataType();

		return inputs[0].getDataType();
	}

	public static ValueType getImplicitValueType(Hop... inputs) {
		ValueType out = null;
		for (int i = 0; i < inputs.length; i++) {
			switch (inputs[i].getValueType()) {
				case FP64:
					return inputs[i].getValueType();
				case FP32:
					out = inputs[i].getValueType();
					break;
				case INT64:
					out = implicitValueType(out, ValueType.INT64);
					break;
				case INT32:
					out = implicitValueType(out, ValueType.INT32);
					break;
				case BOOLEAN:
					out = implicitValueType(out, ValueType.BOOLEAN);
					break;
			}
		}

		return out == null ? inputs[0].getValueType() : out;
	}

	private static ValueType implicitValueType(ValueType type1, ValueType type2) {
		int rank1 = getTypeRank(type1);
		int rank2 = getTypeRank(type2);

		if (rank1 == Integer.MIN_VALUE && rank2 == Integer.MIN_VALUE)
			return null;

		return rank1 > rank2 ? type1 : type2;
	}

	private static int getTypeRank(ValueType vt) {
		if (vt == null)
			return Integer.MIN_VALUE;

		switch (vt) {
			case FP64:
				return 5;
			case FP32:
				return 4;
			case INT64:
				return 3;
			case INT32:
				return 2;
			case BOOLEAN:
				return 1;
		}

		return Integer.MIN_VALUE;
	}
	
	public static AggUnaryOp createSum( Hop input ) {
		return createAggUnaryOp(input, AggOp.SUM, Direction.RowCol);
	}
	
	public static AggUnaryOp createAggUnaryOp( Hop input, String op ) {
		return createAggUnaryOp(input,
			InstructionUtils.getAggOp(op),
			InstructionUtils.getAggDirection(op));
	}
	
	public static AggUnaryOp createAggUnaryOp( Hop input, AggOp op, Direction dir ) {
		DataType dt = (dir==Direction.RowCol) ? DataType.SCALAR : input.getDataType();
		AggUnaryOp auop = new AggUnaryOp(input.getName(), dt, input.getValueType(), op, dir, input);
		auop.setBlocksize(input.getBlocksize());
		copyLineNumbers(input, auop);
		auop.refreshSizeInformation();
		
		return auop;
	}
	
	public static AggBinaryOp createTsmm(Hop input, boolean left) {
		Hop trans = createTranspose(input);
		return createMatrixMultiply(
			left ? trans : input, left ? input : trans);
	}
	
	public static AggBinaryOp createMatrixMultiply(Hop left, Hop right) {
		AggBinaryOp mmult = new AggBinaryOp(left.getName(), left.getDataType(), left.getValueType(), OpOp2.MULT, AggOp.SUM, left, right);
		mmult.setBlocksize(left.getBlocksize());
		copyLineNumbers(left, mmult);
		mmult.refreshSizeInformation();
		return mmult;
	}
	
	public static ParameterizedBuiltinOp createParameterizedBuiltinOp(Hop input, LinkedHashMap<String,Hop> args, ParamBuiltinOp op) {
		DataType dt = op == ParamBuiltinOp.TOSTRING ? DataType.SCALAR : DataType.MATRIX;
		ValueType vt = op == ParamBuiltinOp.TOSTRING ? ValueType.STRING : ValueType.FP64;
		ParameterizedBuiltinOp pbop = new ParameterizedBuiltinOp("tmp", dt, vt, op, args);
		pbop.setBlocksize(input.getBlocksize());
		copyLineNumbers(input, pbop);
		pbop.refreshSizeInformation();
		return pbop;
	}
	
	public static Hop createScalarIndexing(Hop input, long rix, long cix) {
		Hop ix = createIndexingOp(input, rix, cix);
		return createUnary(ix, OpOp1.CAST_AS_SCALAR);
	}
	
	public static IndexingOp createIndexingOp(Hop input, Hop batchsize) {
		LiteralOp rl = new LiteralOp(1);
		LiteralOp cl = new LiteralOp(1);
		Hop cu = createUnary(input, OpOp1.NCOL);
		return createIndexingOp(input, rl, batchsize, cl, cu);
	}
	
	public static IndexingOp createIndexingOp(Hop input, long rix, long cix) {
		LiteralOp row = new LiteralOp(rix);
		LiteralOp col = new LiteralOp(cix);
		return createIndexingOp(input, row, row, col, col);
	}
	
	public static IndexingOp createIndexingOp(Hop input, long rl, long ru, long cl, long cu) {
		return createIndexingOp(input, new LiteralOp(rl), new LiteralOp(ru), new LiteralOp(cl), new LiteralOp(cu));
	}
	
	public static IndexingOp createIndexingOp(Hop input, Hop rl, Hop ru, Hop cl, Hop cu) {
		IndexingOp ix = new IndexingOp("tmp", DataType.MATRIX, ValueType.FP64, input, rl, ru, cl, cu, rl==ru, cl==cu);
		ix.setBlocksize(input.getBlocksize());
		copyLineNumbers(input, ix);
		ix.refreshSizeInformation();
		return ix;
	}
	
	public static LeftIndexingOp createLeftIndexingOp(Hop lhs, Hop rhs, Hop rl, Hop ru, Hop cl, Hop cu) {
		LeftIndexingOp ix = new LeftIndexingOp("tmp", DataType.MATRIX, ValueType.FP64, lhs, rhs, rl, ru, cl, cu, rl==ru, cl==cu);
		ix.setBlocksize(lhs.getBlocksize());
		copyLineNumbers(lhs, ix);
		ix.refreshSizeInformation();
		return ix;
	}
	
	public static NaryOp createNary(OpOpN op, Hop... inputs) {
		Hop mainInput = inputs[0];
		// safe for unordered inputs of Scalars and Matrices
		// e.g.: S*M*S = M
		// safe for Scalar with different value type
		// e.g.: Scalar(Int) * Scalar(FP64) = Scalar(FP64)
		boolean containsMatrix = Arrays.stream(inputs).anyMatch(Hop::isMatrix);
		boolean containsFP64 = Arrays.stream(inputs).anyMatch(h -> h.getValueType() == ValueType.FP64);
		DataType dtOut = containsMatrix ? DataType.MATRIX : mainInput.getDataType();
		ValueType vtOut = containsFP64? ValueType.FP64 : mainInput.getValueType();
		NaryOp nop = new NaryOp(mainInput.getName(), dtOut, vtOut, op, inputs);
		nop.setBlocksize(mainInput.getBlocksize());
		copyLineNumbers(mainInput, nop);
		nop.refreshSizeInformation();
		return nop;
	}
	
	public static Hop createValueHop( Hop hop, boolean row ) {
		Hop ret = null;
		if( row ){
			ret = hop.rowsKnown() ? new LiteralOp(hop.getDim1()) : 
				new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT64, OpOp1.NROW, hop);
		}
		else{
			ret = hop.colsKnown() ? new LiteralOp(hop.getDim2()) :
				new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT64, OpOp1.NCOL, hop);
		}
		
		return ret;
	}
	

	public static DataGenOp createSeqDataGenOp( Hop input ) {
		return createSeqDataGenOp(input, true);
	}
	
	public static DataGenOp createSeqDataGenOp( Hop input, boolean asc ) {
		Hop to = input.rowsKnown() ? new LiteralOp(input.getDim1()) : 
			new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT64, OpOp1.NROW, input);
		if( asc )
			return createSeqDataGenOp(input, new LiteralOp(1), to, new LiteralOp(1));
		else
			return createSeqDataGenOp(input, to, new LiteralOp(1), new LiteralOp(-1));
	}
	
	public static DataGenOp createSeqDataGenOp(Hop proxy, Hop from, Hop to, Hop incr) {
		HashMap<String, Hop> params = new HashMap<>();
		params.put(Statement.SEQ_FROM, from);
		params.put(Statement.SEQ_TO, to);
		params.put(Statement.SEQ_INCR, incr);
		
		//note internal refresh size information
		DataGenOp datagen = new DataGenOp(OpOpDG.SEQ, new DataIdentifier("tmp"), params);
		datagen.setBlocksize(proxy.getBlocksize());
		copyLineNumbers(proxy, datagen);
		
		return datagen;
	}
	
	public static TernaryOp createTernary(Hop mleft, Hop smid, Hop mright, String opcode) {
		return createTernary(mleft, smid, mright, OpOp3.valueOfByOpcode(opcode));
	}
	
	public static TernaryOp createTernary(Hop mleft, Hop smid, Hop mright, OpOp3 op) {
		//NOTe: for ifelse it's sufficient to check mright as smid==mright
		DataType dt = (op == OpOp3.IFELSE) ? mright.getDataType() : DataType.MATRIX;
		ValueType vt = (op == OpOp3.IFELSE) ? mright.getValueType() : ValueType.FP64;
		TernaryOp ternOp = new TernaryOp("tmp", dt, vt, op, mleft, smid, mright);
		ternOp.setBlocksize(Math.max(mleft.getBlocksize(), mright.getBlocksize()));
		copyLineNumbers(mleft, ternOp);
		ternOp.refreshSizeInformation();
		return ternOp;
	}
	
	public static TernaryOp createTernary(Hop in1, Hop in2, Hop in3, Hop in4, Hop in5, OpOp3 op) {
		TernaryOp ternOp = new TernaryOp("tmp", DataType.MATRIX, ValueType.FP64, op, in1, in2, in3, in4, in5, new LiteralOp(true));
		ternOp.setBlocksize(Math.max(in1.getBlocksize(), in2.getBlocksize()));
		copyLineNumbers(in1, ternOp);
		ternOp.refreshSizeInformation();
		return ternOp;
	}
	
	public static Hop createComputeNnz(Hop input) {
		//nnz = sum(A != 0) -> later rewritten to meta-data operation
		return createSum(createBinary(input, new LiteralOp(0), OpOp2.NOTEQUAL));
	}
	
	public static void setOutputParameters( Hop hop, long rlen, long clen, int blen, long nnz ) {
		hop.setDim1(rlen);
		hop.setDim2(clen);
		hop.setBlocksize(blen);
		hop.setNnz(nnz);
	}
	
	public static void setOutputParametersForScalar( Hop hop ) {
		hop.setDataType(DataType.SCALAR);
		hop.setDim1(0);
		hop.setDim2(0);
		hop.setBlocksize(-1);
		hop.setNnz(-1);
	}
	
	public static void refreshOutputParameters( Hop hnew, Hop hold ) {
		hnew.setDim1( hold.getDim1() );
		hnew.setDim2( hold.getDim2() );
		hnew.setBlocksize(hold.getBlocksize());
		hnew.refreshSizeInformation();
	}

	public static void copyLineNumbers(Hop src, Hop dest) {
		dest.setParseInfo(src);
	}
	
	public static void copyLineNumbers(MatrixBlock mb, Hop tread) {
		tread.setBeginLine(1);
		tread.setEndLine(mb.getNumRows());
		tread.setBeginColumn(1);
		tread.setEndColumn(mb.getNumColumns());
	}
	
	public static void copyLineNumbers(MatrixObject mo, Hop tread) {
		tread.setBeginLine(1);
		tread.setEndLine((int)mo.getNumRows());
		tread.setBeginColumn(1);
		tread.setEndColumn((int)mo.getNumColumns());
	}

	public static void updateHopCharacteristics( Hop hop, int blen, Hop src ) {
		updateHopCharacteristics(hop, blen, new MemoTable(), src);
	}
	
	public static void updateHopCharacteristics( Hop hop, int blen, MemoTable memo, Hop src ) {
		//update block sizes and dimensions  
		hop.setBlocksize(blen);
		hop.refreshSizeInformation();
		
		//compute memory estimates (for exec type selection)
		hop.computeMemEstimate(memo);
		
		//update line numbers 
		HopRewriteUtils.copyLineNumbers(src, hop);
	}
	
	///////////////////////////////////
	// hop size information
	
	public static boolean isDimsKnown( Hop hop ) {
		return hop.dimsKnown();
	}
	
	public static boolean isEmpty( Hop hop ) {
		return ( hop.getNnz()==0 );
	}
	
	public static boolean isEqualMatrixSize(BinaryOp hop) {
		return hop.getDataType().isMatrix()
			&& hop.getInput().get(0).getDataType().isMatrix()
			&& hop.getInput().get(1).getDataType().isMatrix()
			&& isEqualSize(hop.getInput().get(0), hop.getInput().get(1));
	}
	
	public static boolean isEqualSize( Hop hop1, Hop hop2 ) {
		return (hop1.dimsKnown() && hop2.dimsKnown()
			&& hop1.getDim1() == hop2.getDim1()
			&& hop1.getDim2() == hop2.getDim2());
	}
	
	public static boolean isEqualSize( Hop hop1, Hop... hops ) {
		boolean ret = hop1.dimsKnown();
		for( int i=0; i<hops.length && ret; i++ )
			ret &= isEqualSize(hop1, hops[i]);
		return ret;	
	}
	
	public static boolean isSingleBlock( Hop hop ) {
		return isSingleBlock(hop, true)
			&& isSingleBlock(hop, false);
	}
	
	
	/**
	 * Checks our BLOCKSIZE CONSTRAINT, w/ awareness of forced single node
	 * execution mode.
	 * 
	 * @param hop high-level operator
	 * @param cols true if cols
	 * @return true if single block
	 */
	public static boolean isSingleBlock( Hop hop, boolean cols )
	{
		//awareness of forced exec single node (e.g., standalone), where we can 
		//guarantee a single block independent of the size because always in CP.
		if( DMLScript.getGlobalExecMode() == ExecMode.SINGLE_NODE ) {
			return true;
		}
		
		//check row- or column-wise single block constraint 
		return cols ? (hop.colsKnown() && hop.getDim2()<=hop.getBlocksize()) :
			(hop.rowsKnown() && hop.getDim1()<=hop.getBlocksize());
	}
	
	public static boolean isOuterProductLikeMM( Hop hop ) {
		return isMatrixMultiply(hop) && hop.dimsKnown() 
			&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(1).dimsKnown()	
			&& hop.getInput().get(0).getDim1() > hop.getInput().get(0).getDim2()
			&& hop.getInput().get(1).getDim1() < hop.getInput().get(1).getDim2();
	}
	
	public static boolean isOuterBinary( Hop hop ) {
		return hop instanceof BinaryOp && ((BinaryOp) hop).isOuter();
	}
	
	public static boolean isValidOuterBinaryOp( OpOp2 op ) {
		return op.isValidOuter();
	}
	
	public static boolean isSparse(Hop hop) {
		return hop.dimsKnown(true) //dims and nnz known
			&& MatrixBlock.evalSparseFormatInMemory(hop.getDim1(), hop.getDim2(), hop.getNnz());
	}
	
	public static boolean isDense(Hop hop) {
		return hop.dimsKnown(true) //dims and nnz known
			&& !MatrixBlock.evalSparseFormatInMemory(hop.getDim1(), hop.getDim2(), hop.getNnz());
	}
	
	public static boolean isSparse( Hop hop, double threshold ) {
		return hop.getSparsity() < threshold;
	}

	public static boolean isEqualValue( Hop hop1, Hop hop2 ) {
		return isEqualValue((LiteralOp)hop1, (LiteralOp)hop2);
	}
	
	public static boolean isEqualValue( LiteralOp hop1, LiteralOp hop2 ) {
		//check for string (no defined double value)
		if( hop1.getValueType()==ValueType.STRING 
			|| hop2.getValueType()==ValueType.STRING ) {
			return hop1.getStringValue()
				.equals(hop2.getStringValue());
		}
		return getDoubleValue(hop1)
			== getDoubleValue(hop2);
	}
	
	public static boolean isNotMatrixVectorBinaryOperation( Hop hop )
	{
		boolean ret = true;
		
		if( hop instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp) hop;
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			boolean mv = (left.getDim1()>1 && right.getDim1()==1)
					|| (left.getDim2()>1 && right.getDim2()==1);
			ret = isDimsKnown(bop) && !mv;
		}
		
		return ret;
	}

	public static boolean isReorg(Hop hop, ReOrgOp type) {
		return hop instanceof ReorgOp && ((ReorgOp)hop).getOp()==type;
	}
	
	public static boolean isReorg(Hop hop, ReOrgOp... types) {
		return ( hop instanceof ReorgOp 
			&& ArrayUtils.contains(types, ((ReorgOp) hop).getOp()));
	}
	
	public static boolean isTransposeOperation(Hop hop) {
		return isReorg(hop, ReOrgOp.TRANS);
	}
	
	public static boolean isTransposeOperation(Hop hop, int maxParents) {
		return isTransposeOperation(hop) && hop.getParent().size() <= maxParents;
	}
	
	public static boolean containsTransposeOperation(List<Hop> hops) {
		boolean ret = false;
		for( Hop hop : hops )
			ret |= isTransposeOperation(hop);
		return ret;
	}
	
	public static boolean isTransposeOfItself(Hop hop1, Hop hop2) {
		return isTransposeOperation(hop1) && hop1.getInput().get(0) == hop2
			|| isTransposeOperation(hop2) && hop2.getInput().get(0) == hop1;
	}
	
	public static boolean isTsmm(Hop input) {
		if( isMatrixMultiply(input) && isTransposeOfItself(
			input.getInput().get(0), input.getInput().get(1)) )
			return true;
		return false;
	}
	
	public static boolean isTsmmInput(Hop input) {
		if( input.getParent().size()==2 )
			for(int i=0; i<2; i++)
				if( isMatrixMultiply(input.getParent().get(i)) && isTransposeOfItself(
					input.getParent().get(i).getInput().get(0), input.getParent().get(i).getInput().get(1)) )
					return true;
		return false;
	}
	
	public static boolean isBinary(Hop hop, OpOp2 type) {
		return hop instanceof BinaryOp && ((BinaryOp)hop).getOp()==type;
	}
	
	public static boolean isBinary(Hop hop, OpOp2... types) {
		return ( hop instanceof BinaryOp 
			&& ArrayUtils.contains(types, ((BinaryOp) hop).getOp()));
	}
	
	public static boolean isBinary(Hop hop, OpOp2 type, int maxParents) {
		return isBinary(hop, type) && hop.getParent().size() <= maxParents;
	}
	
	public static boolean isBinaryPPred(Hop hop) {
		return hop instanceof BinaryOp && ((BinaryOp) hop).isPPredOperation();
	}
	
	public static boolean isBinarySparseSafe(Hop hop) {
		if( !(hop instanceof BinaryOp) )
			return false;
		if( isBinary(hop, OpOp2.MULT, OpOp2.DIV) )
			return true;
		BinaryOp bop = (BinaryOp) hop;
		Hop lit = bop.getInput().get(0) instanceof LiteralOp ? bop.getInput().get(0) :
			bop.getInput().get(1) instanceof LiteralOp ? bop.getInput().get(1) : null;
		return lit != null && OptimizerUtils
			.isBinaryOpSparsityConditionalSparseSafe(bop.getOp(), (LiteralOp)lit);
	}
	
	public static boolean isBinaryMatrixScalarOperation(Hop hop) {
		return hop instanceof BinaryOp && 
			((hop.getInput().get(0).getDataType().isMatrix() && hop.getInput().get(1).getDataType().isScalar())
			||(hop.getInput().get(1).getDataType().isMatrix() && hop.getInput().get(0).getDataType().isScalar()));
	}
	
	public static boolean isBinaryMatrixMatrixOperation(Hop hop) {
		return hop instanceof BinaryOp 
			&& hop.getInput().get(0).getDataType().isMatrix() && hop.getInput().get(1).getDataType().isMatrix()
			&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(0).getDim1() > 1 && hop.getInput().get(0).getDim2() > 1
			&& hop.getInput().get(1).dimsKnown() && hop.getInput().get(1).getDim1() > 1 && hop.getInput().get(1).getDim2() > 1;
	}
	
	public static boolean isBinaryMatrixMatrixOperationWithSharedInput(Hop hop) {
		boolean ret = isBinaryMatrixMatrixOperation(hop);
		ret = ret && (rContainsInput(hop.getInput().get(0), hop.getInput().get(1), new HashSet<Long>())
				|| rContainsInput(hop.getInput().get(1), hop.getInput().get(0), new HashSet<Long>()));
		return ret;
	}
	
	public static boolean isBinaryMatrixScalar(Hop hop, OpOp2 type, double val) {
		return isBinary(hop, type)
			&& (isLiteralOfValue(hop.getInput().get(0), val)
			|| isLiteralOfValue(hop.getInput().get(1), val));
	}
	
	public static boolean isTernary(Hop hop, OpOp3 type) {
		return hop instanceof TernaryOp && ((TernaryOp)hop).getOp()==type;
	}
	
	public static boolean isTernary(Hop hop, OpOp3... types) {
		return ( hop instanceof TernaryOp 
			&& ArrayUtils.contains(types, ((TernaryOp) hop).getOp()));
	}
	
	public static boolean containsInput(Hop current, Hop probe) {
		return rContainsInput(current, probe, new HashSet<Long>());	
	}
	
	private static boolean rContainsInput(Hop current, Hop probe, HashSet<Long> memo) {
		if( memo.contains(current.getHopID()) )
			return false;
		boolean ret = false;
		for( int i=0; i<current.getInput().size() && !ret; i++ )
			ret |= rContainsInput(current.getInput().get(i), probe, memo);
		ret |= (current == probe);
		memo.add(current.getHopID());
		return ret;
	}
	
	public static boolean isData(Hop hop, OpOpData... types) {
		boolean ret = false;
		for( OpOpData type : types )
			ret |= isData(hop, type);
		return ret;
	}
	
	public static boolean isData(Hop hop, OpOpData type) {
		return hop instanceof DataOp && ((DataOp)hop).getOp()==type;
	}
	
	public static boolean isData(Hop hop, OpOpData type, DataType dt) {
		return isData(hop, type) && hop.getDataType()==dt;
	}
	
	public static boolean isTransformEncode(Hop hop){
		return hop instanceof FunctionOp 
			&& (((FunctionOp)hop).getFunctionName().equalsIgnoreCase(Opcodes.TRANSFORMENCODE.toString()));
	}

	public static boolean isBinaryMatrixColVectorOperation(Hop hop) {
		return hop instanceof BinaryOp 
			&& hop.getInput().get(0).getDataType().isMatrix() && hop.getInput().get(1).getDataType().isMatrix()
			&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(1).dimsKnown() && hop.getInput().get(1).getDim2() == 1;
	}

	public static boolean isBinaryMatrixRowVectorOperation(Hop hop) {
		return hop instanceof BinaryOp 
			&& hop.getInput().get(0).getDataType().isMatrix() && hop.getInput().get(1).getDataType().isMatrix()
			&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(1).dimsKnown() && hop.getInput().get(1).getDim1() == 1;
	}
	
	public static boolean isUnary(Hop hop, OpOp1 type) {
		return hop instanceof UnaryOp && ((UnaryOp)hop).getOp()==type;
	}
	
	public static boolean isUnary(Hop hop, OpOp1 type, int maxParents) {
		return isUnary(hop, type) && hop.getParent().size() <= maxParents;
	}
	
	public static boolean isUnary(Hop hop, OpOp1... types) {
		return ( hop instanceof UnaryOp 
			&& ArrayUtils.contains(types, ((UnaryOp) hop).getOp()));
	}

	/**
	 * Check if given hop is of a terminal type.
	 * Terminal hops are either of type print or write.
	 * @param hop for which the type is checked
	 * @return true if hop is terminal
	 */
	public static boolean isTerminalHop(Hop hop){
		return isUnary(hop, OpOp1.PRINT)
			|| isNary(hop, OpOpN.PRINTF)
			|| isData(hop, OpOpData.PERSISTENTWRITE)
			|| isData(hop, OpOpData.TRANSIENTWRITE)
			|| hop instanceof FunctionOp;
	}
	
	public static boolean isMatrixMultiply(Hop hop) {
		return hop instanceof AggBinaryOp && ((AggBinaryOp)hop).isMatrixMultiply();
	}
	
	public static boolean isAggUnaryOp(Hop hop, AggOp op, Direction dir) {
		return isAggUnaryOp(hop, op) && ((AggUnaryOp)hop).getDirection()==dir;
	}
	
	public static boolean isAggUnaryOp(Hop hop, AggOp...op) {
		if( !(hop instanceof AggUnaryOp) )
			return false;
		AggOp hopOp = ((AggUnaryOp)hop).getOp();
		return ArrayUtils.contains(op, hopOp);
	}
	
	public static boolean isSum(Hop hop) {
		return (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getOp()==AggOp.SUM);
	}
	
	public static boolean isSumSq(Hop hop) {
		return (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getOp()==AggOp.SUM_SQ);
	}

	public static boolean isParameterizedBuiltinOp(Hop hop, ParamBuiltinOp type) {
		return hop instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp) hop).getOp().equals(type);
	}

	public static boolean isParameterizedBuiltinOp(Hop hop, ParamBuiltinOp... types) {
		return hop instanceof ParameterizedBuiltinOp && 
			ArrayUtils.contains(types, ((ParameterizedBuiltinOp) hop).getOp());
	}
	
	public static boolean isRemoveEmpty(Hop hop, boolean rows) {
		return isParameterizedBuiltinOp(hop, ParamBuiltinOp.RMEMPTY)
			&& HopRewriteUtils.isLiteralOfValue(
				((ParameterizedBuiltinOp)hop).getParameterHop("margin"), rows?"rows":"cols");
	}

	public static boolean isRemoveEmpty(Hop hop) {
		return isParameterizedBuiltinOp(hop, ParamBuiltinOp.RMEMPTY);
	}
	
	public static boolean isNary(Hop hop, OpOpN type) {
		return hop instanceof NaryOp && ((NaryOp)hop).getOp()==type;
	}
	
	public static boolean isNary(Hop hop, OpOpN... types) {
		return ( hop instanceof NaryOp 
			&& ArrayUtils.contains(types, ((NaryOp) hop).getOp()));
	}
	
	public static boolean isDnn(Hop hop, OpOpDnn type) {
		return hop instanceof DnnOp && ((DnnOp)hop).getOp()==type;
	}
	
	public static boolean isDnn(Hop hop, OpOpDnn... types) {
		return ( hop instanceof DnnOp 
			&& ArrayUtils.contains(types, ((DnnOp) hop).getOp()));
	}
	
	public static boolean isNonZeroIndicator(Hop pred, Hop hop ) {
		return ( pred instanceof BinaryOp && ((BinaryOp)pred).getOp()==OpOp2.NOTEQUAL
			&& pred.getInput().get(0) == hop //depend on common subexpression elimination
			&& pred.getInput().get(1) instanceof LiteralOp
			&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred.getInput().get(1))==0 );
	}

	public static boolean checkInputDataTypes(Hop hop, DataType... dt) {
		for( int i=0; i<hop.getInput().size(); i++ )
			if( hop.getInput().get(i).getDataType() != dt[i] )
				return false;
		return true;
	}
	
	public static boolean checkAvgRowsGteCols(List<Hop> list) {
		if( list.isEmpty() )
			return false;
		double avg = list.stream().mapToDouble(h -> h.getDim1()).sum();
		return (avg/list.size()) >= list.get(0).getDim2();
	}
	
	public static boolean checkConsistentRows(List<Hop> list1, List<Hop> list2) {
		if( list1.size() != list2.size() )
			return false;
		boolean ret = true;
		int len = list1.size();
		for( int i=0; i<len; i++ )
			ret &= list1.get(i).getDim1() == list2.get(i).getDim1();
		return ret;
	}
	
	public static boolean isColumnRightIndexing(Hop hop) {
		return hop instanceof IndexingOp
			&& ((IndexingOp) hop).isColLowerEqualsUpper()
			&& ((hop.dimsKnown() && hop.getDim1() == hop.getInput().get(0).getDim1())
			|| (isLiteralOfValue(hop.getInput().get(1), 1) 
				&& isUnary(hop.getInput().get(2), OpOp1.NROW) 
				&& hop.getInput().get(2).getInput().get(0)==hop.getInput().get(0)));
	}
	
	public static boolean isFullColumnIndexing(LeftIndexingOp hop) {
		return hop.isColLowerEqualsUpper()
			&& isLiteralOfValue(hop.getInput().get(2), 1)
			&& (isLiteralOfValue(hop.getInput().get(3), hop.getDim1())
				|| isSizeExpressionOf(hop.getInput().get(3), hop.getInput().get(0), true));
	}
	
	public static boolean isFullColumnIndexing(IndexingOp hop) {
		return hop.isColLowerEqualsUpper()
			&& isLiteralOfValue(hop.getInput().get(1), 1)
			&& (isLiteralOfValue(hop.getInput().get(2), hop.getDim1())
				|| isSizeExpressionOf(hop.getInput().get(2), hop.getInput().get(0), true));
	}
	
	public static boolean isFullRowIndexing(LeftIndexingOp hop) {
		return hop.isRowLowerEqualsUpper()
			&& isLiteralOfValue(hop.getInput().get(4), 1)
			&& (isLiteralOfValue(hop.getInput().get(5), hop.getDim2())
				|| isSizeExpressionOf(hop.getInput().get(5), hop.getInput().get(0), false));
	}
	
	public static boolean isFullRowIndexing(IndexingOp hop) {
		return hop.isRowLowerEqualsUpper()
			&& isLiteralOfValue(hop.getInput().get(3), 1)
			&& (isLiteralOfValue(hop.getInput().get(4), hop.getDim2())
				|| isSizeExpressionOf(hop.getInput().get(4), hop.getInput().get(0), false));
	}
	
	public static boolean isColumnRangeIndexing(IndexingOp hop) {
		return ((isLiteralOfValue(hop.getInput().get(1), 1)
			&& isLiteralOfValue(hop.getInput().get(2), hop.getInput().get(0).getDim1()))
			|| hop.getDim1() == hop.getInput().get(0).getDim1())
			&& isLiteralOfValue(hop.getInput().get(3), 1)
			&& hop.getInput().get(4) instanceof LiteralOp;
	}
	
	public static boolean isConsecutiveIndex(Hop index, Hop index2) {
		return (index instanceof LiteralOp && index2 instanceof LiteralOp) ?
			getDoubleValueSafe((LiteralOp)index2) == (getDoubleValueSafe((LiteralOp)index)+1) :
			(isBinaryMatrixScalar(index2, OpOp2.PLUS, 1) && 
				(index2.getInput().get(0) == index || index2.getInput().get(1) == index));
	}
	
	public static boolean isUnnecessaryRightIndexing(Hop hop) {
		if( !(hop instanceof IndexingOp) || hop.isScalar() )
			return false;
		//note: in addition to equal sizes, we also check a valid
		//starting row and column ranges of 1 in order to guard against
		//invalid modifications in the presence of invalid index ranges
		//(e.g., X[,2] on a column vector needs to throw an error)
		return ((IndexingOp)hop).isAllRowsAndCols()
			|| (isEqualSize(hop, hop.getInput().get(0))
				&& !(hop.getDim1()==1 && hop.getDim2()==1)
				&& isLiteralOfValue(hop.getInput().get(1), 1)  //rl
				&& isLiteralOfValue(hop.getInput().get(3), 1)); //cl
	}
	
	public static boolean isScalarMatrixBinaryMult( Hop hop ) {
		return hop instanceof BinaryOp && ((BinaryOp)hop).getOp()==OpOp2.MULT
			&& ((hop.getInput().get(0).getDataType()==DataType.SCALAR && hop.getInput().get(1).getDataType()==DataType.MATRIX)
			|| (hop.getInput().get(0).getDataType()==DataType.MATRIX && hop.getInput().get(1).getDataType()==DataType.SCALAR));
	}
	
	public static boolean isBasic1NSequence(Hop hop) {
		if( hop instanceof DataGenOp && ((DataGenOp)hop).getOp() == OpOpDG.SEQ  ) {
			DataGenOp dgop = (DataGenOp) hop;
			Hop from = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_FROM));
			Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
			return (from instanceof LiteralOp && getDoubleValueSafe((LiteralOp)from)==1)
				&&(incr instanceof LiteralOp && getDoubleValueSafe((LiteralOp)incr)==1);
		}
		return false;
	}
	
	public static boolean isBasic1NSequence(Hop seq, Hop input, boolean row) {
		if( seq instanceof DataGenOp && ((DataGenOp)seq).getOp() == OpOpDG.SEQ  ) {
			DataGenOp dgop = (DataGenOp) seq;
			Hop from = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_FROM));
			Hop to = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_TO));
			Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
			return isLiteralOfValue(from, 1) && isLiteralOfValue(incr, 1)
				&& isSizeExpressionOf(to, input, row);
		}
		return false;
	}
	
	public static boolean isBasicN1Sequence(Hop hop)
	{
		boolean ret = false;
		
		if( hop instanceof DataGenOp )
		{
			DataGenOp dgop = (DataGenOp) hop;
			if( dgop.getOp() == OpOpDG.SEQ ){
				Hop to = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_TO));
				Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
				ret = (to instanceof LiteralOp && getDoubleValueSafe((LiteralOp)to)==1)
					&&(incr instanceof LiteralOp && getDoubleValueSafe((LiteralOp)incr)==-1);
			}
		}
		
		return ret;
	}


	public static Hop getBasic1NSequenceMax(Hop hop) {
		if( isDataGenOp(hop, OpOpDG.SEQ) ) {
			DataGenOp dgop = (DataGenOp) hop;
			return dgop.getInput()
				.get(dgop.getParamIndex(Statement.SEQ_TO));
		}
		throw new HopsException("Failed to retrieve 'to' argument from basic 1-N sequence.");
	}
	
	public static boolean isSizeExpressionOf(Hop size, Hop input, boolean row) {
		return (input.dimsKnown() && isLiteralOfValue(size, row?input.getDim1():input.getDim2()))
			|| ((row ? isUnary(size, OpOp1.NROW) : isUnary(size, OpOp1.NCOL)) && (size.getInput().get(0)==input 
			|| (isColumnRightIndexing(input) && size.getInput().get(0)==input.getInput().get(0))));
	}
	
	public static boolean hasOnlyWriteParents( Hop hop, boolean inclTransient, boolean inclPersistent ) {
		boolean ret = true;
		List<Hop> parents = hop.getParent();
		for( Hop p : parents ) {
			if( inclTransient && inclPersistent )
				ret &= ( p instanceof DataOp && (((DataOp)p).getOp()==OpOpData.TRANSIENTWRITE
				|| ((DataOp)p).getOp()==OpOpData.PERSISTENTWRITE));
			else if(inclTransient)
				ret &= ( p instanceof DataOp && ((DataOp)p).getOp()==OpOpData.TRANSIENTWRITE);
			else if(inclPersistent)
				ret &= ( p instanceof DataOp && ((DataOp)p).getOp()==OpOpData.PERSISTENTWRITE);
		}
		return ret;
	}
	
	public static boolean hasOnlyUnaryBinaryParents(Hop hop, boolean disallowLhs) {
		boolean ret = true;
		for( Hop p : hop.getParent() )
			ret &= (p instanceof UnaryOp || (p instanceof BinaryOp 
				&& (!disallowLhs || p.getInput().get(1)==hop)));
		return ret;
	}
	
	public static boolean alwaysRequiresReblock(Hop hop) {
		return (hop instanceof DataOp
			&& ((DataOp)hop).getOp()==OpOpData.PERSISTENTREAD
			 && ((DataOp)hop).getFileFormat()!=FileFormat.BINARY);
	}
	
	public static boolean containsOp(List<Hop> candidates, Class<? extends Hop> clazz) {
		if( candidates != null )
			for( Hop cand : candidates )
				if( cand.getClass().equals(clazz) )
					return true;
		return false;
	}
	
	public static boolean rHasSimpleReadChain(Hop root, String var)
	{
		if( root.isVisited() )
			return false;

		boolean ret = false;
		
		//handle leaf node for variable
		if( root instanceof DataOp && ((DataOp)root).isRead()
			&& root.getName().equals(var) )
		{
			ret = (root.getParent().size()<=1);
		}
		
		//recursively process childs (on the entire path to var, all
		//intermediates are supposed to have at most one consumer, but
		//side-ways inputs can have arbitrary dag structures)
		for( Hop c : root.getInput() ) {
			if( rHasSimpleReadChain(c, var) )
				ret |= root.getParent().size()<=1;
		}
		
		root.setVisited();
		return ret;
	}
	
	public static boolean rContainsRead(Hop root, String var, boolean includeMetaOp)
	{
		if( root.isVisited() )
			return false;

		boolean ret = false;
		
		//handle leaf node for variable
		if( root instanceof DataOp && ((DataOp)root).isRead()
			&& root.getName().equals(var) )
		{
			boolean onlyMetaOp = true;
			if( !includeMetaOp ){
				for( Hop p : root.getParent() ) {
					onlyMetaOp &= (p instanceof UnaryOp 
							&& (((UnaryOp)p).getOp()==OpOp1.NROW 
							|| ((UnaryOp)p).getOp()==OpOp1.NCOL) ); 
				}
				ret = !onlyMetaOp;
			}
			else
				ret = true;
		}
		
		//recursively process childs
		for( Hop c : root.getInput() )
			ret |= rContainsRead(c, var, includeMetaOp);
		
		root.setVisited();
		return ret;
	}
	
	public static Hop createPartialTsmmCbind(Hop X, Hop deltaX, Hop tsmmIn1) {
		//partial rewrite to rewrite tsmm(cbind(in1, in2)) into form that can reuse tsmm(in1)
		// cell bottomLeft = t(lastCol) %*% oldMatrix
		ReorgOp tLastCol = HopRewriteUtils.createTranspose(deltaX);
		AggBinaryOp bottomLeft = HopRewriteUtils.createMatrixMultiply(tLastCol, X);
		// cell topRight = t(oldMatrix) %*% lastCol = t(bottomLeft)
		ReorgOp topRight = HopRewriteUtils.createTranspose(bottomLeft);
		// bottomRight = t(lastCol) %*% lastCol
		AggBinaryOp bottomRight = HopRewriteUtils.createMatrixMultiply(tLastCol, deltaX);
		// rowOne = cbind(lastRes, topRight)
		BinaryOp rowOne = HopRewriteUtils.createBinary(tsmmIn1, topRight, OpOp2.CBIND);
		// rowTwo = cbind(bottomLeft, bottomRight)
		BinaryOp rowTwo = HopRewriteUtils.createBinary(bottomLeft, bottomRight, OpOp2.CBIND);
		// rbind(rowOne, rowTwo)
		return HopRewriteUtils.createBinary(rowOne, rowTwo, OpOp2.RBIND);
	}
	
	//////////////////////////////////////
	// utils for lookup tables
	
	public static boolean isValidOp( AggOp input, AggOp... validTab ) {
		return ArrayUtils.contains(validTab, input);
	}
	
	public static boolean isValidOp( OpOp1 input, OpOp1... validTab ) {
		return ArrayUtils.contains(validTab, input);
	}
	
	public static boolean isValidOp( OpOp2 input, OpOp2... validTab ) {
		return ArrayUtils.contains(validTab, input);
	}
	
	public static boolean isValidOp( ReOrgOp input, ReOrgOp... validTab ) {
		return ArrayUtils.contains(validTab, input);
	}
	
	public static boolean isValidOp( ParamBuiltinOp input, ParamBuiltinOp... validTab ) {
		return ArrayUtils.contains(validTab, input);
	}
	
	public static int getValidOpPos( OpOp2 input, OpOp2... validTab ) {
		return ArrayUtils.indexOf(validTab, input);
	}
	
	/**
	 * Compares the size of outputs from hop1 and hop2, in terms of number
	 * of matrix cells. 
	 * 
	 * @param hop1 high-level operator 1
	 * @param hop2 high-level operator 2
	 * @return 0 if sizes are equal, &lt;0 for hop1&lt;hop2, &gt;0 for hop1&gt;hop2.
	 */
	public static int compareSize( Hop hop1, Hop hop2 ) {
		long size1 = hop1.getDim1() * hop1.getDim2();
		long size2 = hop2.getDim1() * hop2.getDim2();
		return Long.compare(size1, size2);
	}
	
	public static boolean isLastLevelStatementBlock(StatementBlock sb) {
		return !(sb instanceof FunctionStatementBlock 
			|| sb instanceof WhileStatementBlock
			|| sb instanceof IfStatementBlock
			|| sb instanceof ForStatementBlock); //incl parfor
	}
	
	public static boolean isLoopStatementBlock(StatementBlock sb) {
		return sb instanceof WhileStatementBlock
			|| sb instanceof ForStatementBlock; //incl parfor
	}

	// Return true if this loop contains only basic blocks
	public static boolean isLastLevelLoopStatementBlock (StatementBlock sb) {
		if (!isLoopStatementBlock(sb))
			return false;
		if (sb instanceof WhileStatementBlock) {
			WhileStatement wstmt = (WhileStatement) sb.getStatement(0);
			if (wstmt.getBody().isEmpty())
				return false;
			for(StatementBlock csb : wstmt.getBody())
				if (!isLastLevelStatementBlock(csb))
					return false;
		}
		else if (sb instanceof ForStatementBlock) {
			ForStatement fstmt = (ForStatement) sb.getStatement(0);
			if (fstmt.getBody().isEmpty())
				return false;
			for(StatementBlock csb : fstmt.getBody())
				if(!isLastLevelStatementBlock(csb))
					return false;
		}
		return true;
	}
	
	public static long getMaxNrowInput(Hop hop) {
		return getMaxInputDim(hop, true);
	}
	
	public static long getMaxNcolInput(Hop hop) {
		return getMaxInputDim(hop, false);
	}
	
	public static long getMaxInputDim(Hop hop, boolean dim1) {
		return hop.getInput().stream().mapToLong(
			h -> (dim1 ? h.getDim1() : h.getDim2())).max().orElse(-1);
	}
	
	public static long getSumValidInputDims(Hop hop, boolean dim1) {
		if( !hasValidInputDims(hop, dim1) )
			return -1;
		return hop.getInput().stream().mapToLong(
			h -> (dim1 ? h.getDim1() : h.getDim2())).sum();
	}
	
	public static boolean hasValidInputDims(Hop hop, boolean dim1) {
		return hop.getInput().stream().allMatch(
			h -> dim1 ? h.rowsKnown() : h.colsKnown());
	}
	
	public static long getSumValidInputNnz(Hop hop) {
		if( !hasValidInputNnz(hop) )
			return -1;
		return hop.getInput().stream().mapToLong(h -> h.getNnz()).sum();
	}
	
	public static boolean hasValidInputNnz(Hop hop) {
		return hop.getInput().stream().allMatch(h -> h.getNnz() >= 0);
	}
	
	public static long getMaxInputDim(DataCharacteristics[] dc, boolean dim1) {
		return Arrays.stream(dc).mapToLong(
			h -> (dim1 ? h.getRows() : h.getRows())).max().orElse(-1);
	}
	
	public static long getSumValidInputDims(DataCharacteristics[] mc, boolean dim1) {
		if( !hasValidInputDims(mc, dim1) )
			return -1;
		return Arrays.stream(mc).mapToLong(
			h -> (dim1 ? h.getRows() : h.getCols())).sum();
	}
	
	public static boolean hasValidInputDims(DataCharacteristics[] mc, boolean dim1) {
		return Arrays.stream(mc).allMatch(
			h -> dim1 ? h.rowsKnown() : h.colsKnown());
	}
	
	public static long getSumValidInputNnz(DataCharacteristics[] mc, boolean worstcase) {
		if( !hasValidInputNnz(mc, worstcase) )
			return -1;
		return Arrays.stream(mc).mapToLong(h -> h.nnzKnown() ?
			h.getNonZeros() : h.getLength()).sum();
	}
	
	public static boolean hasValidInputNnz(DataCharacteristics[] mc, boolean worstcase) {
		return Arrays.stream(mc).allMatch(h -> h.nnzKnown() || (worstcase && h.dimsKnown()));
	}
	
	public static boolean hasListInputs(Hop hop) {
		return hop.getInput()!= null 
			&& hop.getInput().stream().anyMatch(h -> h.getDataType().isList());
	}
	
	public static boolean containsSecondOrderBuiltin(List<Hop> roots) {
		Hop.resetVisitStatus(roots);
		return roots.stream().anyMatch(r -> containsSecondOrderBuiltin(r));
	}
	
	private static boolean containsSecondOrderBuiltin(Hop hop) {
		if( hop.isVisited() ) return false;
		hop.setVisited();
		return HopRewriteUtils.isNary(hop, OpOpN.EVAL)
			|| (HopRewriteUtils.isParameterizedBuiltinOp(hop, ParamBuiltinOp.PARAMSERV) 
				&& !knownParamservFunctions(hop))
			|| hop.getInput().stream().anyMatch(c -> containsSecondOrderBuiltin(c));
	}
	
	public static boolean knownParamservFunctions(Hop hop) {
		ParameterizedBuiltinOp pop = (ParameterizedBuiltinOp) hop;
		return pop.getParameterHop("upd") instanceof LiteralOp
			&& pop.getParameterHop("agg") instanceof LiteralOp
			&& (pop.getParameterHop("val") == null 
			 || pop.getParameterHop("val") instanceof LiteralOp);
	}
	
	public static boolean knownParamservFunctions(Hop hop, DMLProgram prog) {
		if( !knownParamservFunctions(hop) )
			return false;
		try {
			ParameterizedBuiltinOp pop = (ParameterizedBuiltinOp) hop;
			String supd = ((LiteralOp)pop.getParameterHop("upd")).getStringValue();
			String sagg = ((LiteralOp)pop.getParameterHop("agg")).getStringValue();
			//if functions not existing, let runtime handle it consistently
			return prog.getFunctionStatementBlock(supd) != null
				&& prog.getFunctionStatementBlock(sagg) != null;
		}
		catch(Exception ex) {
			// If the function keys are incorrect this exception is caught for robustness in error messages for users.
			// Intensionally only catching the exception!
			// For debugging if for some reason the error we encountered was something else we LOG the error.
			LOG.error(ex);
			return false;
		}
	}

	public static void setUnoptimizedFunctionCalls(StatementBlock sb) {
		if( sb instanceof FunctionStatementBlock ) {
			FunctionStatement fstmt = (FunctionStatement) sb.getStatement(0);
			for( StatementBlock c : fstmt.getBody() )
				setUnoptimizedFunctionCalls(c);
		}
		else if( sb instanceof IfStatementBlock ) {
			IfStatement stmt = (IfStatement) sb.getStatement(0);
			for( StatementBlock c : stmt.getIfBody() )
				setUnoptimizedFunctionCalls(c);
			for( StatementBlock c : stmt.getElseBody() )
				setUnoptimizedFunctionCalls(c);
		}
		else if( sb instanceof WhileStatementBlock ) {
			WhileStatement stmt = (WhileStatement) sb.getStatement(0);
			for( StatementBlock c : stmt.getBody() )
				setUnoptimizedFunctionCalls(c);
		}
		else if( sb instanceof ForStatementBlock ) { //incl parfor
			ForStatement stmt = (ForStatement) sb.getStatement(0);
			for( StatementBlock c : stmt.getBody() )
				setUnoptimizedFunctionCalls(c);
		}
		else {
			for( Hop root : sb.getHops() )
				if( root instanceof FunctionOp )
					((FunctionOp)root).setCallOptimized(false);
		}
	}
}
