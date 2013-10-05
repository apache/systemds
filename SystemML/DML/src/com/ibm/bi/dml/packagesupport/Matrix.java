/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.packagesupport;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.FSDataInputStream;
import org.nimble.hadoop.HDFSFileManager;

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExternalFunctionProgramBlockCP;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;

/**
 * Class to represent the matrix input type
 * 
 * 
 * 
 */
public class Matrix extends FIO 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		super(Type.Matrix);
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
		
		if( _mo != null ) //CP ext function
		{
			try
			{
				MatrixBlock mb = _mo.acquireRead();
				ret = DataConverter.convertToDoubleMatrix( mb );
				_mo.release();
			}
			catch(Exception ex)
			{
				throw new PackageRuntimeException(ex);
			}
		}
		else //traditional ext function (matrix file produced by reblock)
		{
			double[][] arr = new double[(int) _rows][(int) _cols];
	
			try {
	
				String[] files;
	
				// get file names for this matrix.
	
				files = HDFSFileManager.getFileNamesWithPrefixStatic(this
						.getFilePath() + "/");
				String line = null;
	
				// read each file into memory.
				for (String file : files) {
	
					FSDataInputStream inStrm;
					inStrm = HDFSFileManager.getInputStreamStatic(file);
					BufferedReader br = new BufferedReader(new InputStreamReader(
							inStrm));
	
					while ((line = br.readLine()) != null) {
	
						StringTokenizer tk = new StringTokenizer(line);
						int i, j;
						double val;
	
						i = Integer.parseInt(tk.nextToken());
						j = Integer.parseInt(tk.nextToken());
						val = Double.parseDouble(tk.nextToken());
	
						arr[i - 1][j - 1] = val;
					}
				}
			} catch (Exception e) {
				throw new PackageRuntimeException(e.toString());
			}
			ret = arr;
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