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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.hops.DataGenOp;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.FileFormatTypes;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.Hop.OpOp3;
import org.apache.sysml.hops.Hop.ParamBuiltinOp;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.LeftIndexingOp;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.MemoTable;
import org.apache.sysml.hops.ParameterizedBuiltinOp;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.hops.TernaryOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.UtilFunctions;

public class HopRewriteUtils 
{

	public static boolean isValueTypeCast( OpOp1 op )
	{
		return (   op == OpOp1.CAST_AS_BOOLEAN 
				|| op == OpOp1.CAST_AS_INT 
				|| op == OpOp1.CAST_AS_DOUBLE );
	}
	
	//////////////////////////////////
	// literal handling

	public static boolean getBooleanValue( LiteralOp op )
		throws HopsException
	{
		switch( op.getValueType() )
		{
			case DOUBLE:  return op.getDoubleValue() != 0; 
			case INT:	  return op.getLongValue()   != 0;
			case BOOLEAN: return op.getBooleanValue();
			
			default: throw new HopsException("Invalid boolean value: "+op.getValueType());
		}
	}

	public static boolean getBooleanValueSafe( LiteralOp op )
	{
		try
		{
			switch( op.getValueType() )
			{
				case DOUBLE:  return op.getDoubleValue() != 0; 
				case INT:	  return op.getLongValue()   != 0;
				case BOOLEAN: return op.getBooleanValue();
				
				default: throw new HopsException("Invalid boolean value: "+op.getValueType());
			}
		}
		catch(Exception ex){
			//silently ignore error
		}
		
		return false;
	}

	public static double getDoubleValue( LiteralOp op )
		throws HopsException
	{
		switch( op.getValueType() )
		{
			case DOUBLE:  return op.getDoubleValue(); 
			case INT:	  return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			
			default: throw new HopsException("Invalid double value: "+op.getValueType());
		}
	}
	
	public static double getDoubleValueSafe( LiteralOp op )
	{
		try
		{
			switch( op.getValueType() )
			{
				case DOUBLE:  return op.getDoubleValue(); 
				case INT:	  return op.getLongValue();
				case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
				
				default: throw new HopsException("Invalid double value: "+op.getValueType());
			}
		}
		catch(Exception ex){
			//silently ignore error
		}
		
		return Double.MAX_VALUE;
	}
	
	/**
	 * Return the int value of a LiteralOp (as a long).
	 *
	 * Note: For comparisons, this is *only* to be used in situations
	 * in which the value is absolutely guaranteed to be an integer.
	 * Otherwise, a safer alternative is `getDoubleValue`.
	 * 
	 * @param op literal operator
	 * @return long value of literator op
	 * @throws HopsException if HopsException occurs
	 */
	public static long getIntValue( LiteralOp op )
		throws HopsException
	{
		switch( op.getValueType() )
		{
			case DOUBLE:  return UtilFunctions.toLong(op.getDoubleValue()); 
			case INT:	  return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			
			default: throw new HopsException("Invalid int value: "+op.getValueType());
		}
	}
	
	public static long getIntValueSafe( LiteralOp op )
	{
		try
		{
			switch( op.getValueType() )
			{
				case DOUBLE:  return UtilFunctions.toLong(op.getDoubleValue()); 
				case INT:	  return op.getLongValue();
				case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
				default:
					throw new RuntimeException("Invalid int value: "+op.getValueType());
			}
		}
		catch(Exception ex){
			//silently ignore error
		}
		
		return Long.MAX_VALUE;
	}
	
	public static boolean isLiteralOfValue( Hop hop, double val ) {
		return (hop instanceof LiteralOp 
			&& (hop.getValueType()==ValueType.DOUBLE || hop.getValueType()==ValueType.INT)
			&& getDoubleValueSafe((LiteralOp)hop)==val);
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
		ArrayList<Hop> parents = hold.getParent();
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
	
	public static Hop createDataGenOp( Hop input, double value ) 
		throws HopsException
	{		
		Hop rows = (input.getDim1()>0) ? new LiteralOp(input.getDim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, input);
		Hop cols = (input.getDim2()>0) ? new LiteralOp(input.getDim2()) :
			       new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT, OpOp1.NCOL, input);
		Hop val = new LiteralOp(value);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_LAMBDA, new LiteralOp(-1.0));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp(1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
		datagen.setOutputBlocksizes(input.getRowsInBlock(), input.getColsInBlock());
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
	 * @throws HopsException if HopsException occurs
	 */
	public static DataGenOp copyDataGenOp( DataGenOp inputGen, double scale, double shift ) 
		throws HopsException
	{		
		HashMap<String, Integer> params = inputGen.getParamIndexMap();
		Hop rows = inputGen.getInput().get(params.get(DataExpression.RAND_ROWS));
		Hop cols = inputGen.getInput().get(params.get(DataExpression.RAND_COLS));
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
		
		HashMap<String, Hop> params2 = new HashMap<String, Hop>();
		params2.put(DataExpression.RAND_ROWS, rows);
		params2.put(DataExpression.RAND_COLS, cols);
		params2.put(DataExpression.RAND_MIN, sminHop);
		params2.put(DataExpression.RAND_MAX, smaxHop);
		params2.put(DataExpression.RAND_PDF, pdf);
		params2.put(DataExpression.RAND_LAMBDA, mean);
		params2.put(DataExpression.RAND_SPARSITY, sparsity);		
		params2.put(DataExpression.RAND_SEED, seed );
		
		//note internal refresh size information
		DataGenOp datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params2);
		datagen.setOutputBlocksizes(inputGen.getRowsInBlock(), inputGen.getColsInBlock());
		copyLineNumbers(inputGen, datagen);
		
		if( smin==0 && smax==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOp( Hop rowInput, Hop colInput, double value ) 
		throws HopsException
	{		
		Hop rows = (rowInput.getDim1()>0) ? new LiteralOp(rowInput.getDim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, rowInput);
		Hop cols = (colInput.getDim2()>0) ? new LiteralOp(colInput.getDim2()) :
			       new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT, OpOp1.NCOL, colInput);
		Hop val = new LiteralOp(value);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_LAMBDA, new LiteralOp(-1.0));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp(1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
		datagen.setOutputBlocksizes(rowInput.getRowsInBlock(), colInput.getColsInBlock());
		copyLineNumbers(rowInput, datagen);
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOp( Hop rowInput, boolean tRowInput, Hop colInput, boolean tColInput, double value ) 
		throws HopsException
	{		
		long nrow = tRowInput ? rowInput.getDim2() : rowInput.getDim1();
		long ncol = tColInput ? colInput.getDim1() : rowInput.getDim2();
		
		Hop rows = (nrow>0) ? new LiteralOp(nrow) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, tRowInput?OpOp1.NCOL:OpOp1.NROW, rowInput);
		Hop cols = (ncol>0) ? new LiteralOp(ncol) :
			       new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT, tColInput?OpOp1.NROW:OpOp1.NCOL, colInput);
		Hop val = new LiteralOp(value);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_LAMBDA,new LiteralOp(-1.0));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp(1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
		datagen.setOutputBlocksizes(rowInput.getRowsInBlock(), colInput.getColsInBlock());
		copyLineNumbers(rowInput, datagen);
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOpByVal( Hop rowInput, Hop colInput, double value ) 
		throws HopsException
	{		
		Hop val = new LiteralOp(value);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		params.put(DataExpression.RAND_ROWS, rowInput);
		params.put(DataExpression.RAND_COLS, colInput);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_LAMBDA, new LiteralOp(-1.0));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp(1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
		datagen.setOutputBlocksizes(rowInput.getRowsInBlock(), colInput.getColsInBlock());
		copyLineNumbers(rowInput, datagen);
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static ReorgOp createTranspose(Hop input) {
		return createReorg(input, ReOrgOp.TRANSPOSE);
	}
	
	public static ReorgOp createReorg(Hop input, ReOrgOp rop)
	{
		ReorgOp transpose = new ReorgOp(input.getName(), input.getDataType(), input.getValueType(), rop, input);
		transpose.setOutputBlocksizes(input.getRowsInBlock(), input.getColsInBlock());
		copyLineNumbers(input, transpose);
		transpose.refreshSizeInformation();	
		
		return transpose;
	}
	
	public static UnaryOp createUnary(Hop input, OpOp1 type) 
	{
		DataType dt = (type==OpOp1.CAST_AS_SCALAR) ? DataType.SCALAR : 
			(type==OpOp1.CAST_AS_MATRIX) ? DataType.MATRIX : input.getDataType();
		ValueType vt = (type==OpOp1.CAST_AS_MATRIX) ? ValueType.DOUBLE : input.getValueType();
		UnaryOp unary = new UnaryOp(input.getName(), dt, vt, type, input);
		unary.setOutputBlocksizes(input.getRowsInBlock(), input.getColsInBlock());
		if( type == OpOp1.CAST_AS_SCALAR || type == OpOp1.CAST_AS_MATRIX ) {
			int dim = (type==OpOp1.CAST_AS_SCALAR) ? 0 : 1;
			int blksz = (type==OpOp1.CAST_AS_SCALAR) ? 0 : ConfigurationManager.getBlocksize();
			setOutputParameters(unary, dim, dim, blksz, blksz, -1);		
		}
		
		copyLineNumbers(input, unary);
		unary.refreshSizeInformation();	
		
		return unary;
	}
	
	public static BinaryOp createBinaryMinus(Hop input) {
		return createBinary(new LiteralOp(0), input, OpOp2.MINUS);
	}
	
	public static BinaryOp createBinary(Hop input1, Hop input2, OpOp2 op)
	{
		Hop mainInput = input1.getDataType().isMatrix() ? input1 : 
			input2.getDataType().isMatrix() ? input2 : input1;
		BinaryOp bop = new BinaryOp(mainInput.getName(), mainInput.getDataType(), 
			mainInput.getValueType(), op, input1, input2);
		//cleanup value type for relational operations
		if( bop.isPPredOperation() && bop.getDataType().isScalar() )
			bop.setValueType(ValueType.BOOLEAN);
		bop.setOutputBlocksizes(mainInput.getRowsInBlock(), mainInput.getColsInBlock());
		copyLineNumbers(mainInput, bop);
		bop.refreshSizeInformation();	
		return bop;
	}
	
	public static AggUnaryOp createSum( Hop input ) {
		return createAggUnaryOp(input, AggOp.SUM, Direction.RowCol);
	}
	
	public static AggUnaryOp createAggUnaryOp( Hop input, AggOp op, Direction dir ) {
		DataType dt = (dir==Direction.RowCol) ? DataType.SCALAR : input.getDataType();
		AggUnaryOp auop = new AggUnaryOp(input.getName(), dt, input.getValueType(), op, dir, input);
		auop.setOutputBlocksizes(input.getRowsInBlock(), input.getColsInBlock());
		copyLineNumbers(input, auop);
		auop.refreshSizeInformation();
		
		return auop;
	}
	
	public static AggBinaryOp createMatrixMultiply(Hop left, Hop right) {
		AggBinaryOp mmult = new AggBinaryOp(left.getName(), left.getDataType(), left.getValueType(), OpOp2.MULT, AggOp.SUM, left, right);
		mmult.setOutputBlocksizes(left.getRowsInBlock(), right.getColsInBlock());
		copyLineNumbers(left, mmult);
		mmult.refreshSizeInformation();
		
		return mmult;
	}
	
	public static ParameterizedBuiltinOp createParameterizedBuiltinOp(Hop input, HashMap<String,Hop> args, ParamBuiltinOp op) {
		ParameterizedBuiltinOp pbop = new ParameterizedBuiltinOp("tmp", DataType.MATRIX, ValueType.DOUBLE, op, args);
		pbop.setOutputBlocksizes(input.getRowsInBlock(), input.getColsInBlock());
		copyLineNumbers(input, pbop);
		pbop.refreshSizeInformation();
		
		return pbop;
	}
	
	public static Hop createScalarIndexing(Hop input, long rix, long cix) {
		Hop ix = createMatrixIndexing(input, rix, cix);
		return createUnary(ix, OpOp1.CAST_AS_SCALAR);
	}
	
	public static Hop createMatrixIndexing(Hop input, long rix, long cix) {
		LiteralOp row = new LiteralOp(rix);
		LiteralOp col = new LiteralOp(cix);
		IndexingOp ix = new IndexingOp("tmp", DataType.MATRIX, ValueType.DOUBLE, input, row, row, col, col, true, true);
		ix.setOutputBlocksizes(input.getRowsInBlock(), input.getColsInBlock());
		copyLineNumbers(input, ix);
		ix.refreshSizeInformation();
		return ix;
	}
	
	public static Hop createValueHop( Hop hop, boolean row ) 
		throws HopsException
	{
		Hop ret = null;
		if( row ){
			ret = (hop.getDim1()>0) ? new LiteralOp(hop.getDim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, hop);
		}
		else{
			ret = (hop.getDim2()>0) ? new LiteralOp(hop.getDim2()) :
			       new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT, OpOp1.NCOL, hop);
		}
		
		return ret;
	}
	

	public static DataGenOp createSeqDataGenOp( Hop input ) 
		throws HopsException
	{
		return createSeqDataGenOp(input, true);
	}
	
	public static DataGenOp createSeqDataGenOp( Hop input, boolean asc ) 
		throws HopsException
	{		
		Hop to = (input.getDim1()>0) ? new LiteralOp(input.getDim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, input);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		if( asc ) {
			params.put(Statement.SEQ_FROM, new LiteralOp(1));
			params.put(Statement.SEQ_TO, to);
			params.put(Statement.SEQ_INCR, new LiteralOp(1));
		}
		else {
			params.put(Statement.SEQ_FROM, to);
			params.put(Statement.SEQ_TO, new LiteralOp(1));
			params.put(Statement.SEQ_INCR, new LiteralOp(-1));	
		}
		
		//note internal refresh size information
		DataGenOp datagen = new DataGenOp(DataGenMethod.SEQ, new DataIdentifier("tmp"), params);
		datagen.setOutputBlocksizes(input.getRowsInBlock(), input.getColsInBlock());
		copyLineNumbers(input, datagen);
		
		return datagen;
	}
	
	public static TernaryOp createTernaryOp(Hop mleft, Hop smid, Hop mright, OpOp3 op) {
		TernaryOp ternOp = new TernaryOp("tmp", DataType.MATRIX, ValueType.DOUBLE, op, mleft, smid, mright);
		ternOp.setOutputBlocksizes(mleft.getRowsInBlock(), mleft.getColsInBlock());
		copyLineNumbers(mleft, ternOp);
		ternOp.refreshSizeInformation();
		return ternOp;
	}
	
	public static void setOutputParameters( Hop hop, long rlen, long clen, long brlen, long bclen, long nnz ) {
		hop.setDim1( rlen );
		hop.setDim2( clen );
		hop.setOutputBlocksizes(brlen, bclen );
		hop.setNnz( nnz );
	}
	
	public static void setOutputParametersForScalar( Hop hop ) {
		hop.setDataType(DataType.SCALAR);
		hop.setDim1( 0 );
		hop.setDim2( 0 );
		hop.setOutputBlocksizes(-1, -1 );
		hop.setNnz( -1 );
	}
	
	public static void refreshOutputParameters( Hop hnew, Hop hold ) {
		hnew.setDim1( hold.getDim1() );
		hnew.setDim2( hold.getDim2() );
		hnew.setOutputBlocksizes(hold.getRowsInBlock(), hold.getColsInBlock());
		hnew.refreshSizeInformation();
	}
	
	public static void copyLineNumbers( Hop src, Hop dest ) {
		dest.setAllPositions(src.getBeginLine(), src.getBeginColumn(), src.getEndLine(), src.getEndColumn());
	}
	
	public static void updateHopCharacteristics( Hop hop, long brlen, long bclen, Hop src )
	{
		updateHopCharacteristics(hop, brlen, bclen, new MemoTable(), src);
	}
	
	public static void updateHopCharacteristics( Hop hop, long brlen, long bclen, MemoTable memo, Hop src )
	{
		//update block sizes and dimensions  
		hop.setOutputBlocksizes(brlen, bclen);
		hop.refreshSizeInformation();
		
		//compute memory estimates (for exec type selection)
		hop.computeMemEstimate(memo);
		
		//update line numbers 
		HopRewriteUtils.copyLineNumbers(src, hop);
	}
	
	///////////////////////////////////
	// hop size information
	
	public static boolean isDimsKnown( Hop hop )
	{
		return ( hop.getDim1()>0 && hop.getDim2()>0 );
	}
	
	public static boolean isEmpty( Hop hop )
	{
		return ( hop.getNnz()==0 );
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
		if( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE ) {
			return true;
		}
		
		//check row- or column-wise single block constraint 
		return cols ? (hop.getDim2()>0 && hop.getDim2()<=hop.getColsInBlock())
				    : (hop.getDim1()>0 && hop.getDim1()<=hop.getRowsInBlock());
	}
	
	public static boolean isOuterProductLikeMM( Hop hop ) {
		return isMatrixMultiply(hop) && hop.dimsKnown() 
			&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(1).dimsKnown()	
			&& hop.getInput().get(0).getDim1() > hop.getInput().get(0).getDim2()
			&& hop.getInput().get(1).getDim1() < hop.getInput().get(1).getDim2();
	}
	
	public static boolean isSparse( Hop hop ) {
		return hop.dimsKnown(true) //dims and nnz known
			&& MatrixBlock.evalSparseFormatInMemory(hop.getDim1(), hop.getDim2(), hop.getNnz());
	}
	
	public static boolean isEqualValue( LiteralOp hop1, LiteralOp hop2 ) 
		throws HopsException
	{
		//check for string (no defined double value)
		if(    hop1.getValueType()==ValueType.STRING 
			|| hop2.getValueType()==ValueType.STRING )
		{
			return false;
		}
		
		double val1 = getDoubleValue(hop1);
		double val2 = getDoubleValue(hop2);
		
		return ( val1 == val2 );
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

	public static boolean isTransposeOperation(Hop hop) {
		return (hop instanceof ReorgOp && ((ReorgOp)hop).getOp()==ReOrgOp.TRANSPOSE);
	}
	
	public static boolean isTransposeOperation(Hop hop, int maxParents) {
		return isTransposeOperation(hop) && hop.getParent().size() <= maxParents;
	}
	
	public static boolean containsTransposeOperation(ArrayList<Hop> hops) {
		boolean ret = false;
		for( Hop hop : hops )
			ret |= isTransposeOperation(hop);
		return ret;
	}
	
	public static boolean isTransposeOfItself(Hop hop1, Hop hop2) {
		return isTransposeOperation(hop1) && hop1.getInput().get(0) == hop2
			|| isTransposeOperation(hop2) && hop2.getInput().get(0) == hop1;	
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
	
	public static boolean isBinaryMatrixColVectorOperation(Hop hop) {
		return hop instanceof BinaryOp 
			&& hop.getInput().get(0).getDataType().isMatrix() && hop.getInput().get(1).getDataType().isMatrix()
			&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(1).dimsKnown() && hop.getInput().get(1).getDim2() == 1;
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
	
	public static boolean isMatrixMultiply(Hop hop) {
		return hop instanceof AggBinaryOp && ((AggBinaryOp)hop).isMatrixMultiply();
	}
	
	public static boolean isAggUnaryOp(Hop hop, AggOp...op) {
		if( !(hop instanceof AggUnaryOp) )
			return false;
		AggOp hopOp = ((AggUnaryOp)hop).getOp();
		for( AggOp opi : op ) 
			if( hopOp == opi )
				return true;
		return false; 
	}
	
	public static boolean isSum(Hop hop) {
		return (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getOp()==AggOp.SUM);
	}
	
	public static boolean isSumSq(Hop hop) {
		return (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getOp()==AggOp.SUM_SQ);
	}
	
	public static boolean isNonZeroIndicator(Hop pred, Hop hop )
	{
		if( pred instanceof BinaryOp && ((BinaryOp)pred).getOp()==OpOp2.NOTEQUAL
			&& pred.getInput().get(0) == hop //depend on common subexpression elimination
			&& pred.getInput().get(1) instanceof LiteralOp
			&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred.getInput().get(1))==0 )
		{
			return true;
		}
		
		return false;
	}

	public static boolean checkInputDataTypes(Hop hop, DataType... dt) {
		for( int i=0; i<hop.getInput().size(); i++ )
			if( hop.getInput().get(i).getDataType() != dt[i] )
				return false;
		return true;
	}
	
	public static boolean isFullColumnIndexing(LeftIndexingOp hop)
	{
		boolean colPred = hop.getColLowerEqualsUpper();  //single col
		
		Hop rl = hop.getInput().get(2);
		Hop ru = hop.getInput().get(3);
		
		return colPred && rl instanceof LiteralOp && getDoubleValueSafe((LiteralOp)rl)==1
				&& ru instanceof LiteralOp && getDoubleValueSafe((LiteralOp)ru)==hop.getDim1();
	}
	
	public static boolean isFullRowIndexing(LeftIndexingOp hop)
	{
		boolean rowPred = hop.getRowLowerEqualsUpper();  //single row
		
		Hop cl = hop.getInput().get(4);
		Hop cu = hop.getInput().get(5);
		
		return rowPred && cl instanceof LiteralOp && getDoubleValueSafe((LiteralOp)cl)==1
				&& cu instanceof LiteralOp && getDoubleValueSafe((LiteralOp)cu)==hop.getDim2();
	}
	
	public static boolean isScalarMatrixBinaryMult( Hop hop ) {
		return hop instanceof BinaryOp && ((BinaryOp)hop).getOp()==OpOp2.MULT
			&& ((hop.getInput().get(0).getDataType()==DataType.SCALAR && hop.getInput().get(1).getDataType()==DataType.MATRIX)
			|| (hop.getInput().get(0).getDataType()==DataType.MATRIX && hop.getInput().get(1).getDataType()==DataType.SCALAR));
	}
	
	public static boolean isBasic1NSequence(Hop hop) {
		if( hop instanceof DataGenOp && ((DataGenOp)hop).getOp() == DataGenMethod.SEQ  ) {
			DataGenOp dgop = (DataGenOp) hop;
			Hop from = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_FROM));
			Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
			return (from instanceof LiteralOp && getDoubleValueSafe((LiteralOp)from)==1)
				&&(incr instanceof LiteralOp && getDoubleValueSafe((LiteralOp)incr)==1);
		}
		return false;
	}
	
	public static boolean isBasic1NSequence(Hop seq, Hop input, boolean row) {
		if( seq instanceof DataGenOp && ((DataGenOp)seq).getOp() == DataGenMethod.SEQ  ) {
			DataGenOp dgop = (DataGenOp) seq;
			Hop from = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_FROM));
			Hop to = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_TO));
			Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
			return isLiteralOfValue(from, 1) && isLiteralOfValue(incr, 1)
				&& (isLiteralOfValue(to, row?input.getDim1():input.getDim2())
					|| (to instanceof UnaryOp && ((UnaryOp)to).getOp()==(row?
						OpOp1.NROW:OpOp1.NCOL) && to.getInput().get(0)==input));
		}
		return false;
	}
	
	public static boolean isBasicN1Sequence(Hop hop)
	{
		boolean ret = false;
		
		if( hop instanceof DataGenOp )
		{
			DataGenOp dgop = (DataGenOp) hop;
			if( dgop.getOp() == DataGenMethod.SEQ ){
				Hop to = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_TO));
				Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
				ret = (to instanceof LiteralOp && getDoubleValueSafe((LiteralOp)to)==1)
					&&(incr instanceof LiteralOp && getDoubleValueSafe((LiteralOp)incr)==-1);
			}
		}
		
		return ret;
	}

	public static LiteralOp getBasic1NSequenceMaxLiteral(Hop hop) 
		throws HopsException
	{
		if( hop instanceof DataGenOp )
		{
			DataGenOp dgop = (DataGenOp) hop;
			if( dgop.getOp() == DataGenMethod.SEQ ){
				Hop to = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_TO));
				if( to instanceof LiteralOp )
					return (LiteralOp)to;
			}
		}
		
		throw new HopsException("Failed to retrieve 'to' argument from basic 1-N sequence.");
	}
	
	
	public static boolean hasOnlyWriteParents( Hop hop, boolean inclTransient, boolean inclPersistent )
	{
		boolean ret = true;
		
		ArrayList<Hop> parents = hop.getParent();
		for( Hop p : parents )
		{
			if( inclTransient && inclPersistent )
				ret &= ( p instanceof DataOp && (((DataOp)p).getDataOpType()==DataOpTypes.TRANSIENTWRITE
				|| ((DataOp)p).getDataOpType()==DataOpTypes.PERSISTENTWRITE));
			else if(inclTransient)
				ret &= ( p instanceof DataOp && ((DataOp)p).getDataOpType()==DataOpTypes.TRANSIENTWRITE);
			else if(inclPersistent)
				ret &= ( p instanceof DataOp && ((DataOp)p).getDataOpType()==DataOpTypes.PERSISTENTWRITE);
		}
			
				
		return ret;
	}
	
	public static boolean alwaysRequiresReblock(Hop hop)
	{
		return (    hop instanceof DataOp 
				 && ((DataOp)hop).getDataOpType()==DataOpTypes.PERSISTENTREAD
				 && ((DataOp)hop).getInputFormatType()!=FileFormatTypes.BINARY);
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
	 * of matrix cells. Note that this methods throws a RuntimeException
	 * if either hop has unknown dimensions. 
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
}
