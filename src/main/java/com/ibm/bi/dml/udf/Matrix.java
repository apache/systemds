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

package com.ibm.bi.dml.udf;

import java.io.IOException;

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExternalFunctionProgramBlockCP;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.io.MatrixReader;
import com.ibm.bi.dml.runtime.io.MatrixReaderFactory;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;

/**
 * Class to represent the matrix input type
 * 
 * 
 * 
 */
public class Matrix extends FunctionParameter 
{
	
	private static final long serialVersionUID = -1058329938431848909L;
	
	private String 		 _filePath;
	private long 		 _rows;
	private long 		 _cols;
	private ValueType 	 _vType;
	private MatrixObject _mo;

	public enum ValueType {
		Double, 
		Integer
	};
	
	/**
	 * This constructor invokes Matrix(String path, long rows, long cols, ValueType vType)
	 * with a default filename of ExternalFunctionProgramBlockCP and hence, should only
	 * be used by CP external functions.
	 * 
	 * @param rows
	 * @param cols
	 * @param vType
	 */
	public Matrix(long rows, long cols, ValueType vType) {
		this( ExternalFunctionProgramBlockCP.DEFAULT_FILENAME, 
			  rows, cols, vType );
	}

	/**
	 * Constructor that takes matrix file path, num rows, num cols, and matrix
	 * value type as parameters.
	 * 
	 * @param path
	 * @param rows
	 * @param cols
	 * @param vType
	 */
	public Matrix(String path, long rows, long cols, ValueType vType) {
		super(FunctionParameterType.Matrix);
		_filePath = path;
		_rows = rows;
		_cols = cols;
		_vType = vType;
	}
	
	public void setMatrixObject( MatrixObject mo )
	{
		_mo = mo;
	}
	
	public MatrixObject getMatrixObject()
	{
		return _mo;
	}

	/**
	 * Method to get file path for matrix.
	 * 
	 * @return
	 */
	public String getFilePath() {
		return _filePath;
	}

	/**
	 * Method to get the number of rows in the matrix.
	 * 
	 * @return
	 */
	public long getNumRows() {
		return _rows;
	}

	/**
	 * Method to get the number of cols in the matrix.
	 * 
	 * @return
	 */

	public long getNumCols() {
		return _cols;
	}

	/**
	 * Method to get value type for this matrix.
	 * 
	 * @return
	 */
	public ValueType getValueType() {
		return _vType;
	}

	/**
	 * Method to get matrix as double array. This should only be used if the
	 * user knows the matrix fits in memory. We are using the dense
	 * representation.
	 * 
	 * @return
	 */
	public double[][] getMatrixAsDoubleArray() 
	{
		double[][] ret = null;
		
		try 
		{
			if( _mo != null ) //CP ext function
			{
				MatrixBlock mb = _mo.acquireRead();
				ret = DataConverter.convertToDoubleMatrix( mb );
				_mo.release();
			}
			else //traditional ext function (matrix file produced by reblock)
			{
				MatrixReader reader = MatrixReaderFactory.createMatrixReader(InputInfo.TextCellInputInfo);
				MatrixBlock mb = reader.readMatrixFromHDFS(this.getFilePath(), _rows, _cols, -1, -1, -1);
				ret = DataConverter.convertToDoubleMatrix( mb );
			}
		}
		catch(Exception ex)
		{
			throw new PackageRuntimeException(ex);
		}
			
		return ret;

	}

	/**
	 * Method to set matrix as double array. This should only be used if the
	 * user knows the matrix fits in memory. We are using the dense
	 * representation.
	 * 
	 * @return
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 */
	public void setMatrixDoubleArray(double[][] data /*, OutputInfo oinfo, InputInfo iinfo*/) 
		throws IOException, DMLRuntimeException 
	{
		MatrixBlock mb = DataConverter.convertToMatrixBlock(data);
		setMatrixDoubleArray(mb, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
	}
	
	/**
	 * Method to set matrix as double array. This should only be used if the
	 * user knows the matrix fits in memory. We are using the dense
	 * representation.
	 * 
	 * @return
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 */
	public void setMatrixDoubleArray(double[] data /*, OutputInfo oinfo, InputInfo iinfo*/) 
		throws IOException, DMLRuntimeException 
	{
		MatrixBlock mb = DataConverter.convertToMatrixBlock(data, true);
		setMatrixDoubleArray(mb, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
	}
	
	/**
	 * Method to set matrix as double array. This should only be used if the
	 * user knows the matrix fits in memory. We are using the dense
	 * representation.
	 * 
	 * @return
	 * @throws IOException 
	 */
	public void setMatrixDoubleArray(MatrixBlock mb, OutputInfo oinfo, InputInfo iinfo) 
		throws IOException 
	{
		_rows = mb.getNumRows();
		_cols = mb.getNumColumns();
		long nnz = mb.getNonZeros();
		int rblen = DMLTranslator.DMLBlockSize;
		int cblen = DMLTranslator.DMLBlockSize;
		
		MatrixCharacteristics mc = new MatrixCharacteristics(_rows, _cols, rblen, cblen, nnz);
		MatrixFormatMetaData mfmd = new MatrixFormatMetaData(mc, oinfo, iinfo);
		try 
		{
			_mo = new MatrixObject(Expression.ValueType.DOUBLE, _filePath, mfmd);
			_mo.acquireModify( mb );
			_mo.release();
		} 
		catch(Exception e) 
		{
			throw new IOException(e);
		} 
	}
}