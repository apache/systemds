/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.CTableMap;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

/**
 * 
 * 
 */
public class CTable extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -5374880447194177236L;

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
	public void execute(double v1, double v2, double w, boolean ignoreZeros, CTableMap resultMap) 
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
		
		// skip this entry as it does not fall within specified output dimensions
		if( ignoreZeros && row == 0 && col == 0 ) {
			return;
		}
		
		//check for incorrect ctable inputs
		if( row <= 0 || col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero): "+v1+" "+v2);
		} 
	
		//hash group-by for core ctable computation
		resultMap.aggregate(row, col, w);	
	}	

	/**
	 * 
	 * @param v1
	 * @param v2
	 * @param w
	 * @param ctableResult
	 * @throws DMLRuntimeException
	 */
	public void execute(double v1, double v2, double w, boolean ignoreZeros, MatrixBlock ctableResult) 
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
		
		// skip this entry as it does not fall within specified output dimensions
		if( ignoreZeros && row == 0 && col == 0 ) {
			return;
		}
		
		//check for incorrect ctable inputs
		if( row <= 0 || col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero): "+v1+" "+v2);
		}
		
		// skip this entry as it does not fall within specified output dimensions
		if( row > ctableResult.getNumRows() || col > ctableResult.getNumColumns() ) {
			return;
		}
		
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
