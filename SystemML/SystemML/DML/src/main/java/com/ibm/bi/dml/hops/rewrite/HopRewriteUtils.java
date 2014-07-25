/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.AggOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class HopRewriteUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
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
		switch( op.get_valueType() )
		{
			case DOUBLE:  return op.getDoubleValue() != 0; 
			case INT:	  return op.getLongValue()   != 0;
			case BOOLEAN: return op.getBooleanValue();
			
			default: throw new HopsException("Invalid boolean value: "+op.get_valueType());
		}
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
		switch( op.get_valueType() )
		{
			case DOUBLE:  return op.getDoubleValue(); 
			case INT:	  return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			
			default: throw new HopsException("Invalid double value: "+op.get_valueType());
		}
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
		switch( op.get_valueType() )
		{
			case DOUBLE:  return UtilFunctions.toLong(op.getDoubleValue()); 
			case INT:	  return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			
			default: throw new HopsException("Invalid int value: "+op.get_valueType());
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
			switch( op.get_valueType() )
			{
				case DOUBLE:  return UtilFunctions.toLong(op.getDoubleValue()); 
				case INT:	  return op.getLongValue();
				case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			}
		}
		catch(Exception ex){
			//silently ignore error
		}
		
		return -1;
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
			switch( op.get_valueType() )
			{
				case DOUBLE:  ret = new DoubleObject(op.getDoubleValue()); break;
				case INT:	  ret = new IntObject((int)op.getLongValue()); break;
				case BOOLEAN: ret = new BooleanObject(op.getBooleanValue()); break;
				case STRING:  ret = new StringObject(op.getStringValue()); break;
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
		Hop rows = (input.get_dim1()>0) ? new LiteralOp(String.valueOf(input.get_dim1()),input.get_dim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, input);
		Hop cols = (input.get_dim2()>0) ? new LiteralOp(String.valueOf(input.get_dim2()),input.get_dim2()) :
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
		datagen.set_rows_in_block(input.get_rows_in_block());
		datagen.set_cols_in_block(input.get_cols_in_block());
		
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
		datagen.set_rows_in_block(inputGen.get_rows_in_block());
		datagen.set_cols_in_block(inputGen.get_cols_in_block());
		
		if( smin==0 && smax==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOp( Hop rowInput, Hop colInput, double value ) 
		throws HopsException
	{		
		Hop rows = (rowInput.get_dim1()>0) ? new LiteralOp(String.valueOf(rowInput.get_dim1()),rowInput.get_dim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, rowInput);
		Hop cols = (colInput.get_dim2()>0) ? new LiteralOp(String.valueOf(colInput.get_dim2()),colInput.get_dim2()) :
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
		datagen.set_rows_in_block(rowInput.get_rows_in_block());
		datagen.set_cols_in_block(colInput.get_cols_in_block());
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createDataGenOp( Hop rowInput, boolean tRowInput, Hop colInput, boolean tColInput, double value ) 
		throws HopsException
	{		
		long nrow = tRowInput ? rowInput.get_dim2() : rowInput.get_dim1();
		long ncol = tColInput ? colInput.get_dim1() : rowInput.get_dim2();
		
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
		datagen.set_rows_in_block(rowInput.get_rows_in_block());
		datagen.set_cols_in_block(colInput.get_cols_in_block());
		
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
		datagen.set_rows_in_block(rowInput.get_rows_in_block());
		datagen.set_cols_in_block(colInput.get_cols_in_block());
		
		if( value==0 )
			datagen.setNnz(0);
			
		return datagen;
	}
	
	public static Hop createValueHop( Hop hop, boolean row ) 
		throws HopsException
	{
		Hop ret = null;
		if( row ){
			ret = (hop.get_dim1()>0) ? new LiteralOp(String.valueOf(hop.get_dim1()),hop.get_dim1()) : 
			       new UnaryOp("tmprows", DataType.SCALAR, ValueType.INT, OpOp1.NROW, hop);
		}
		else{
			ret = (hop.get_dim2()>0) ? new LiteralOp(String.valueOf(hop.get_dim2()),hop.get_dim2()) :
			       new UnaryOp("tmpcols", DataType.SCALAR, ValueType.INT, OpOp1.NCOL, hop);
		}
		
		return ret;
	}
	
	public static void setOutputParameters( Hop hop, long rlen, long clen, long brlen, long bclen, long nnz )
	{
		hop.set_dim1( rlen );
		hop.set_dim2( clen );
		hop.set_rows_in_block( brlen );
		hop.set_cols_in_block( bclen );
		hop.setNnz( nnz );
	}
	
	public static void refreshOutputParameters( Hop hnew, Hop hold )
	{
		hnew.set_dim1( hold.get_dim1() );
		hnew.set_dim2( hold.get_dim2() );
		hnew.set_rows_in_block(hold.get_rows_in_block());
		hnew.set_cols_in_block(hold.get_cols_in_block());
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
		return ( hop.get_dim1()>0 && hop.get_dim2()>0 );
	}
	
	public static boolean isEmpty( Hop hop )
	{
		return ( hop.getNnz()==0 );
	}
	
	public static boolean isEqualSize( Hop hop1, Hop hop2 )
	{
		return (   hop1.get_dim1() == hop2.get_dim1()
				&& hop1.get_dim2() == hop2.get_dim2());
	}
	
	public static boolean isEqualValue( LiteralOp hop1, LiteralOp hop2 ) 
		throws HopsException
	{
		//check for string (no defined double value)
		if(    hop1.get_valueType()==ValueType.STRING 
			|| hop2.get_valueType()==ValueType.STRING )
		{
			return false;
		}
		
		double val1 = getDoubleValue(hop1);
		double val2 = getDoubleValue(hop2);
		
		return ( val1 == val2 );
	}
	
	
	public static boolean hasOnlyTransientWriteParents( Hop hop )
	{
		boolean ret = true;
		
		ArrayList<Hop> parents = hop.getParent();
		for( Hop p : parents )
			ret &= ( p instanceof DataOp && ((DataOp)p).get_dataop()==DataOpTypes.TRANSIENTWRITE );
				
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
	
}
