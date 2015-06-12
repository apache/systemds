/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.AggBinaryOp;
import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.AggOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.Hop.ReOrgOp;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.ReorgOp;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.cp.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.cp.IntObject;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.cp.StringObject;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class HopRewriteUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";


	/**
	 * 
	 * @param op
	 * @return
	 */
	public static boolean isValueTypeCast( OpOp1 op )
	{
		return (   op == OpOp1.CAST_AS_BOOLEAN 
				|| op == OpOp1.CAST_AS_INT 
				|| op == OpOp1.CAST_AS_DOUBLE );
	}
	
	//////////////////////////////////
	// literal handling
	
	/**
	 * 
	 * @param op
	 * @return
	 * @throws HopsException
	 */
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
	
	/**
	 * 
	 * @param op
	 * @return
	 * @throws HopsException
	 */
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
	
	/**
	 * 
	 * @param op
	 * @return
	 */
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
	 * 
	 * @param op
	 * @return
	 * @throws HopsException
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
	
	/**
	 * 
	 * @param op
	 * @return
	 */
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
	
	/**
	 * 
	 * @param op
	 * @return
	 * @throws HopsException
	 */
	public static ScalarObject getScalarObject( LiteralOp op )
	{
		ScalarObject ret = null;
		
		try
		{
			switch( op.getValueType() )
			{
				case DOUBLE:  ret = new DoubleObject(op.getDoubleValue()); break;
				case INT:	  ret = new IntObject(op.getLongValue()); break;
				case BOOLEAN: ret = new BooleanObject(op.getBooleanValue()); break;
				case STRING:  ret = new StringObject(op.getStringValue()); break;
				default:
					throw new DMLRuntimeException("Invalid scalar object value type: "+op.getValueType());
			}
		}
		catch(Exception ex)
		{
			throw new RuntimeException("Failed to create scalar object for constant. Continue.", ex);
		}
		
		return ret;
	}

	///////////////////////////////////
	// hop dag transformations
	
	

	public static int getChildReferencePos( Hop parent, Hop child )
	{
		ArrayList<Hop> childs = parent.getInput();
		return childs.indexOf(child);
	}
	
	public static void removeChildReference( Hop parent, Hop child )
	{
		//remove child reference
		parent.getInput().remove( child );
		child.getParent().remove( parent );
	}
	
	public static void removeChildReferenceByPos( Hop parent, Hop child, int posChild )
	{
		//remove child reference
		parent.getInput().remove( posChild );
		child.getParent().remove( parent );
	}
	
	public static void removeChildReferenceByPos( Hop parent, Hop child, int posChild, int posParent )
	{
		//remove child reference
		parent.getInput().remove( posChild );
		child.getParent().remove( posParent );
	}
	
	public static void removeAllChildReferences( Hop parent )
	{
		//remove parent reference from all childs
		for( Hop child : parent.getInput() )
			child.getParent().remove(parent);
		
		//remove all child references
		parent.getInput().clear();
	}
	
	public static void addChildReference( Hop parent, Hop child )
	{
		parent.getInput().add( child );
		child.getParent().add( parent );
	}
	
	public static void addChildReference( Hop parent, Hop child, int pos )
	{
		parent.getInput().add( pos, child );
		child.getParent().add( parent );
	}
	
	/**
	 * 
	 * @param input
	 * @param value
	 * @return
	 * @throws HopsException
	 */
	public static Hop createDataGenOp( Hop input, double value ) 
		throws HopsException
	{		
		Hop rows = (input.getDim1()>0) ? new LiteralOp(String.valueOf(input.getDim1()),input.getDim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, input);
		Hop cols = (input.getDim2()>0) ? new LiteralOp(String.valueOf(input.getDim2()),input.getDim2()) :
			       new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT, OpOp1.NCOL, input);
		Hop val = new LiteralOp(String.valueOf(value), value);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM,DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp("1.0",1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(String.valueOf(DataGenOp.UNSPECIFIED_SEED),DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
		datagen.setRowsInBlock(input.getRowsInBlock());
		datagen.setColsInBlock(input.getColsInBlock());
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	/**
	 * Assumes that min and max are literal ops, needs to be checked from outside.
	 * 
	 * @param inputGen
	 * @param scale
	 * @param intercept
	 * @return
	 * @throws HopsException
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
		
		Hop sminHop = new LiteralOp(String.valueOf(smin), smin);
		Hop smaxHop = new LiteralOp(String.valueOf(smax), smax);
		
		HashMap<String, Hop> params2 = new HashMap<String, Hop>();
		params2.put(DataExpression.RAND_ROWS, rows);
		params2.put(DataExpression.RAND_COLS, cols);
		params2.put(DataExpression.RAND_MIN, sminHop);
		params2.put(DataExpression.RAND_MAX, smaxHop);
		params2.put(DataExpression.RAND_PDF, pdf);
		params2.put(DataExpression.RAND_SPARSITY, sparsity);		
		params2.put(DataExpression.RAND_SEED, seed );
		
		//note internal refresh size information
		DataGenOp datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params2);
		datagen.setRowsInBlock(inputGen.getRowsInBlock());
		datagen.setColsInBlock(inputGen.getColsInBlock());
		
		if( smin==0 && smax==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOp( Hop rowInput, Hop colInput, double value ) 
		throws HopsException
	{		
		Hop rows = (rowInput.getDim1()>0) ? new LiteralOp(String.valueOf(rowInput.getDim1()),rowInput.getDim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, rowInput);
		Hop cols = (colInput.getDim2()>0) ? new LiteralOp(String.valueOf(colInput.getDim2()),colInput.getDim2()) :
			       new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT, OpOp1.NCOL, colInput);
		Hop val = new LiteralOp(String.valueOf(value), value);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM,DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp("1.0",1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(String.valueOf(DataGenOp.UNSPECIFIED_SEED),DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
		datagen.setRowsInBlock(rowInput.getRowsInBlock());
		datagen.setColsInBlock(colInput.getColsInBlock());
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOp( Hop rowInput, boolean tRowInput, Hop colInput, boolean tColInput, double value ) 
		throws HopsException
	{		
		long nrow = tRowInput ? rowInput.getDim2() : rowInput.getDim1();
		long ncol = tColInput ? colInput.getDim1() : rowInput.getDim2();
		
		Hop rows = (nrow>0) ? new LiteralOp(String.valueOf(nrow), nrow) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, tRowInput?OpOp1.NCOL:OpOp1.NROW, rowInput);
		Hop cols = (ncol>0) ? new LiteralOp(String.valueOf(ncol), ncol) :
			       new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT, tColInput?OpOp1.NROW:OpOp1.NCOL, colInput);
		Hop val = new LiteralOp(String.valueOf(value), value);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		params.put(DataExpression.RAND_ROWS, rows);
		params.put(DataExpression.RAND_COLS, cols);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM,DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp("1.0",1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(String.valueOf(DataGenOp.UNSPECIFIED_SEED),DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
		datagen.setRowsInBlock(rowInput.getRowsInBlock());
		datagen.setColsInBlock(colInput.getColsInBlock());
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOpByVal( Hop rowInput, Hop colInput, double value ) 
		throws HopsException
	{		
		Hop val = new LiteralOp(String.valueOf(value), value);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		params.put(DataExpression.RAND_ROWS, rowInput);
		params.put(DataExpression.RAND_COLS, colInput);
		params.put(DataExpression.RAND_MIN, val);
		params.put(DataExpression.RAND_MAX, val);
		params.put(DataExpression.RAND_PDF, new LiteralOp(DataExpression.RAND_PDF_UNIFORM,DataExpression.RAND_PDF_UNIFORM));
		params.put(DataExpression.RAND_SPARSITY, new LiteralOp("1.0",1.0));		
		params.put(DataExpression.RAND_SEED, new LiteralOp(String.valueOf(DataGenOp.UNSPECIFIED_SEED),DataGenOp.UNSPECIFIED_SEED) );
		
		//note internal refresh size information
		Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
		datagen.setRowsInBlock(rowInput.getRowsInBlock());
		datagen.setColsInBlock(colInput.getColsInBlock());
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	/**
	 * 
	 * @param input
	 * @return
	 */
	public static ReorgOp createTranspose(Hop input)
	{
		ReorgOp transpose = new ReorgOp(input.getName(), input.getDataType(), input.getValueType(), ReOrgOp.TRANSPOSE, input);
		HopRewriteUtils.setOutputBlocksizes(transpose, input.getRowsInBlock(), input.getColsInBlock());
		HopRewriteUtils.copyLineNumbers(input, transpose);
		transpose.refreshSizeInformation();	
		
		return transpose;
	}
	
	/**
	 * 
	 * @param left
	 * @param right
	 * @return
	 */
	public static AggBinaryOp createMatrixMultiply(Hop left, Hop right)
	{
		AggBinaryOp mmult = new AggBinaryOp(left.getName(), left.getDataType(), left.getValueType(), OpOp2.MULT, AggOp.SUM, left, right);
		mmult.setRowsInBlock(left.getRowsInBlock());
		mmult.setColsInBlock(right.getColsInBlock());
		mmult.refreshSizeInformation();
		
		return mmult;
	}
	
	public static Hop createValueHop( Hop hop, boolean row ) 
		throws HopsException
	{
		Hop ret = null;
		if( row ){
			ret = (hop.getDim1()>0) ? new LiteralOp(String.valueOf(hop.getDim1()),hop.getDim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, hop);
		}
		else{
			ret = (hop.getDim2()>0) ? new LiteralOp(String.valueOf(hop.getDim2()),hop.getDim2()) :
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
		Hop to = (input.getDim1()>0) ? new LiteralOp(String.valueOf(input.getDim1()),input.getDim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, input);
		
		HashMap<String, Hop> params = new HashMap<String, Hop>();
		if( asc ) {
			params.put(Statement.SEQ_FROM, new LiteralOp("1",1));
			params.put(Statement.SEQ_TO, to);
			params.put(Statement.SEQ_INCR, new LiteralOp("1",1));
		}
		else {
			params.put(Statement.SEQ_FROM, to);
			params.put(Statement.SEQ_TO, new LiteralOp("1",1));
			params.put(Statement.SEQ_INCR, new LiteralOp("-1",-1));	
		}
		
		//note internal refresh size information
		DataGenOp datagen = new DataGenOp(DataGenMethod.SEQ, new DataIdentifier("tmp"), params);
		datagen.setRowsInBlock(input.getRowsInBlock());
		datagen.setColsInBlock(input.getColsInBlock());
		
		return datagen;
	}
	
	public static void setOutputBlocksizes( Hop hop, long brlen, long bclen )
	{
		hop.setRowsInBlock( brlen );
		hop.setColsInBlock( bclen );
	}
	
	public static void setOutputParameters( Hop hop, long rlen, long clen, long brlen, long bclen, long nnz )
	{
		hop.setDim1( rlen );
		hop.setDim2( clen );
		hop.setRowsInBlock( brlen );
		hop.setColsInBlock( bclen );
		hop.setNnz( nnz );
	}
	
	public static void setOutputParametersForScalar( Hop hop )
	{
		hop.setDim1( 0 );
		hop.setDim2( 0 );
		hop.setRowsInBlock( -1 );
		hop.setColsInBlock( -1 );
		hop.setNnz( -1 );
	}
	
	public static void refreshOutputParameters( Hop hnew, Hop hold )
	{
		hnew.setDim1( hold.getDim1() );
		hnew.setDim2( hold.getDim2() );
		hnew.setRowsInBlock(hold.getRowsInBlock());
		hnew.setColsInBlock(hold.getColsInBlock());
		hnew.refreshSizeInformation();
	}
	
	public static void copyLineNumbers( Hop src, Hop dest )
	{
		dest.setAllPositions(src.getBeginLine(), src.getBeginColumn(), src.getEndLine(), src.getEndColumn());
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
	
	public static boolean isEqualSize( Hop hop1, Hop hop2 )
	{
		return (   hop1.getDim1() == hop2.getDim1()
				&& hop1.getDim2() == hop2.getDim2());
	}
	
	public static boolean isSingleBlock( Hop hop, boolean cols )
	{
		return cols ? (hop.getDim2()>0 && hop.getDim2()<=hop.getColsInBlock())
				    : (hop.getDim1()>0 && hop.getDim1()<=hop.getRowsInBlock());
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
	
	/**
	 * 
	 * @param hop
	 * @return
	 */
	public static boolean isTransposeOperation(Hop hop)
	{
		return (hop instanceof ReorgOp && ((ReorgOp)hop).getOp()==ReOrgOp.TRANSPOSE);
	}
	
	public static boolean hasOnlyWriteParents( Hop hop, boolean inclTransient, boolean inclPersistent )
	{
		boolean ret = true;
		
		ArrayList<Hop> parents = hop.getParent();
		for( Hop p : parents )
		{
			if( inclTransient && inclPersistent )
				ret &= ( p instanceof DataOp && (((DataOp)p).get_dataop()==DataOpTypes.TRANSIENTWRITE
				|| ((DataOp)p).get_dataop()==DataOpTypes.PERSISTENTWRITE));
			else if(inclTransient)
				ret &= ( p instanceof DataOp && ((DataOp)p).get_dataop()==DataOpTypes.TRANSIENTWRITE);
			else if(inclPersistent)
				ret &= ( p instanceof DataOp && ((DataOp)p).get_dataop()==DataOpTypes.PERSISTENTWRITE);
		}
			
				
		return ret;
	}

	//////////////////////////////////////
	// utils for lookup tables
	
	public static boolean isValidOp( AggOp input, AggOp[] validTab )
	{
		for( AggOp valid : validTab )
			if( valid == input )
				return true;
		return false;
	}
	
	public static boolean isValidOp( OpOp1 input, OpOp1[] validTab )
	{
		for( OpOp1 valid : validTab )
			if( valid == input )
				return true;
		return false;
	}
	
	public static boolean isValidOp( OpOp2 input, OpOp2[] validTab )
	{
		for( OpOp2 valid : validTab )
			if( valid == input )
				return true;
		return false;
	}
	
	public static int getValidOpPos( OpOp2 input, OpOp2[] validTab )
	{
		for( int i=0; i<validTab.length; i++ ) {
			 OpOp2 valid = validTab[i];
			 if( valid == input )
					return i;
		}
		return -1;
	}
	
}
