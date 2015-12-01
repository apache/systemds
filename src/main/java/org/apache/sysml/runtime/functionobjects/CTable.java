/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.functionobjects;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.CTableMap;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.util.UtilFunctions;

/**
 * 
 * 
 */
public class CTable extends ValueFunction 
{

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

	/**
	 * 
	 * @param row
	 * @param v2
	 * @param w
	 * @return
	 * @throws DMLRuntimeException
	 */
	public Pair<MatrixIndexes,Double> execute( long row, double v2, double w ) 
		throws DMLRuntimeException
	{
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v2) || Double.isNaN(w) ) {
			return new Pair<MatrixIndexes,Double>(new MatrixIndexes(-1,-1), w);
		}
		
		// safe casts to long for consistent behavior with indexing
		long col = UtilFunctions.toLong( v2 );
				
		if( col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (value <= zero): "+v2);
		} 
		
		return new Pair<MatrixIndexes,Double>(new MatrixIndexes(row, col), w);
	}
}
