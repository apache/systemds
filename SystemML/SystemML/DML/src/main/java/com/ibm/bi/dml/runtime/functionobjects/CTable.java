/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import java.util.HashMap;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

/**
 * 
 * 
 */
public class CTable extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static CTable singleObj = null;
	
	private CTable() {
		// nothing to do here
	}
	
	public static CTable getCTableFnObject() {
		if ( singleObj == null )
			singleObj = new CTable();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	/**
	 * 
	 * @param v1
	 * @param v2
	 * @param w
	 * @param ctableResult
	 * @throws DMLRuntimeException
	 */
	public void execute(double v1, double v2, HashMap<MatrixIndexes, Double> ctableResult) 
		throws DMLRuntimeException 
	{	
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v1) || Double.isNaN(v2) ) {
			return;
		}
		
		// safe casts to long for consistent behavior with indexing
		long row = UtilFunctions.toLong( v1 );
		long col = UtilFunctions.toLong( v2 );
		
		if ( row <= 0 || col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero): "+v1+" "+v2);
		} 
		
		MatrixIndexes temp=new MatrixIndexes(row, col);
		Double oldw=ctableResult.get(temp);
		if(oldw==null)
			oldw=0.0;
		ctableResult.put(temp, oldw+1);
	}
	
	/**
	 * 
	 * @param v1
	 * @param v2
	 * @param w
	 * @param ctableResult
	 * @throws DMLRuntimeException
	 */
	public void execute(double v1, double v2, MatrixBlock ctableResult) 
		throws DMLRuntimeException 
	{	
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v1) || Double.isNaN(v2) ) {
			return;
		}
		
		// safe casts to long for consistent behavior with indexing
		long row = UtilFunctions.toLong( v1 );
		long col = UtilFunctions.toLong( v2 );
		
		if ( row <= 0 || col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero): "+v1+" "+v2);
		} 
		// skip this entry as it does not fall within specified output dimensions
		if ( row > ctableResult.getNumRows() || col > ctableResult.getNumColumns() )
			return;
		
		ctableResult.addValue((int)row, (int)col, 1);
	}
	
	/**
	 * 
	 * @param v1
	 * @param v2
	 * @param w
	 * @param ctableResult
	 * @throws DMLRuntimeException
	 */
	public void execute(double v1, double v2, double w, HashMap<MatrixIndexes, Double> ctableResult) 
		throws DMLRuntimeException 
	{	
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v1) || Double.isNaN(v2) || Double.isNaN(w) ) {
			return;
		}
		
		// safe casts to long for consistent behavior with indexing
		long row = UtilFunctions.toLong( v1 );
		long col = UtilFunctions.toLong( v2 );
				
		if ( row <= 0 || col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero): "+v1+" "+v2);
		} 
		
		MatrixIndexes temp=new MatrixIndexes(row, col);
		Double oldw=ctableResult.get(temp);
		if(oldw==null)
			oldw=0.0;
		ctableResult.put(temp, oldw+w);
	}	

	/**
	 * 
	 * @param v1
	 * @param v2
	 * @param w
	 * @param ctableResult
	 * @throws DMLRuntimeException
	 */
	public void execute(double v1, double v2, double w, MatrixBlock ctableResult) 
		throws DMLRuntimeException 
	{	
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v1) || Double.isNaN(v2) || Double.isNaN(w) ) {
			return;
		}
		
		// safe casts to long for consistent behavior with indexing
		long row = UtilFunctions.toLong( v1 );
		long col = UtilFunctions.toLong( v2 );
				
		if ( row <= 0 || col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero): "+v1+" "+v2);
		} 
		
		// skip this entry as it does not fall within specified output dimensions
		if ( row > ctableResult.getNumRows() || col > ctableResult.getNumColumns() )
			return;
		
		ctableResult.addValue((int)row-1, (int)col-1, w);
	}
	
	/**
	 * 
	 * @param row
	 * @param v2
	 * @param w
	 * @param maxCol
	 * @return
	 */
	public int execute(int row, double v2, double w, int maxCol, MatrixBlock ctableResult) 
		throws DMLRuntimeException 
	{	
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v2) || Double.isNaN(w) ) {
			return maxCol;
		}
		
		// safe casts to long for consistent behavior with indexing
		long col = UtilFunctions.toLong( v2 );
				
		if( col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (value <= zero): "+v2);
		} 
		
		//set weight as value (expand is guaranteed to address different cells)
		ctableResult.quickSetValue((int)row-1, (int)col-1, w);
		
		//maintain max seen col 
		return Math.max(maxCol, (int)col);
	}

}
