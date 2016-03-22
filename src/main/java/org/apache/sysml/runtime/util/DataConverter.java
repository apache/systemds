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

package org.apache.sysml.runtime.util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.io.MatrixReader;
import org.apache.sysml.runtime.io.MatrixReaderFactory;
import org.apache.sysml.runtime.io.MatrixWriter;
import org.apache.sysml.runtime.io.MatrixWriterFactory;
import org.apache.sysml.runtime.io.ReadProperties;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CTableMap;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.udf.Matrix;


/**
 * This class provides methods to read and write matrix blocks from to HDFS using different data formats.
 * Those functionalities are used especially for CP read/write and exporting in-memory matrices to HDFS
 * (before executing MR jobs).
 * 
 */
public class DataConverter 
{
	
	//////////////
	// READING and WRITING of matrix blocks to/from HDFS
	// (textcell, binarycell, binaryblock)
	///////
	
	/**
	 * 
	 * @param mat
	 * @param dir
	 * @param outputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo,  MatrixCharacteristics mc )
		throws IOException
	{
		writeMatrixToHDFS(mat, dir, outputinfo, mc, -1, null);
	}
	
	/**
	 * 
	 * @param mat
	 * @param dir
	 * @param outputinfo
	 * @param mc
	 * @param replication
	 * @param formatProperties
	 * @throws IOException
	 */
	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo, MatrixCharacteristics mc, int replication, FileFormatProperties formatProperties)
		throws IOException
	{
		try {
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter( outputinfo, replication, formatProperties );
			writer.writeMatrixToHDFS(mat, dir, mc.getRows(), mc.getCols(), mc.getRowsPerBlock(), mc.getColsPerBlock(), mc.getNonZeros());
		}
		catch(Exception e)
		{
			throw new IOException(e);
		}
	}
	
	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen, boolean localFS) 
		throws IOException
	{	
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.localFS = localFS;
		
		//expected matrix is sparse (default SystemML usecase)
		return readMatrixFromHDFS(prop);
	}
	
	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
		throws IOException
	{	
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		
		//expected matrix is sparse (default SystemML usecase)
		return readMatrixFromHDFS(prop);
	}

	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen, double expectedSparsity) 
		throws IOException
	{	
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.expectedSparsity = expectedSparsity;
		
		return readMatrixFromHDFS(prop);
	}

	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @param localFS
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen, double expectedSparsity, boolean localFS) 
		throws IOException
	{
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.expectedSparsity = expectedSparsity;
		prop.localFS = localFS;
		
		return readMatrixFromHDFS(prop);
	}
	
	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @param localFS
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen, double expectedSparsity, FileFormatProperties formatProperties) 
	throws IOException
	{
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.expectedSparsity = expectedSparsity;
		prop.formatProperties = formatProperties;
		
		//prop.printMe();
		return readMatrixFromHDFS(prop);
	}
	
	/**
	 * Core method for reading matrices in format textcell, matrixmarket, binarycell, or binaryblock 
	 * from HDFS into main memory. For expected dense matrices we directly copy value- or block-at-a-time 
	 * into the target matrix. In contrast, for sparse matrices, we append (column-value)-pairs and do a 
	 * final sort if required in order to prevent large reorg overheads and increased memory consumption 
	 * in case of unordered inputs.  
	 * 
	 * DENSE MxN input:
	 *  * best/average/worst: O(M*N)
	 * SPARSE MxN input
	 *  * best (ordered, or binary block w/ clen<=bclen): O(M*N)
	 *  * average (unordered): O(M*N*log(N))
	 *  * worst (descending order per row): O(M * N^2)
	 * 
	 * NOTE: providing an exact estimate of 'expected sparsity' can prevent a full copy of the result
	 * matrix block (required for changing sparse->dense, or vice versa)
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(ReadProperties prop) 
		throws IOException
	{	
		//Timing time = new Timing(true);
		
		long estnnz = (long)(prop.expectedSparsity*prop.rlen*prop.clen);
	
		//core matrix reading 
		MatrixBlock ret = null;
		try {
			MatrixReader reader = MatrixReaderFactory.createMatrixReader(prop);
			ret = reader.readMatrixFromHDFS(prop.path, prop.rlen, prop.clen, prop.brlen, prop.bclen, estnnz);
		}
		catch(DMLRuntimeException rex)
		{
			throw new IOException(rex);
		}	
		
		//System.out.println("read matrix ("+prop.rlen+","+prop.clen+","+ret.getNonZeros()+") in "+time.stop());
				
		return ret;
	}

	
	//////////////
	// Utils for CREATING and COPYING matrix blocks 
	///////
	
	/**
	 * Creates a two-dimensional double matrix of the input matrix block. 
	 * 
	 * @param mb
	 * @return
	 */
	public static double[][] convertToDoubleMatrix( MatrixBlock mb )
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		double[][] ret = new double[rows][cols]; //0-initialized
		
		if( mb.getNonZeros() > 0 )
		{
			if( mb.isInSparseFormat() )
			{
				Iterator<IJV> iter = mb.getSparseBlockIterator();
				while( iter.hasNext() )
				{
					IJV cell = iter.next();
					ret[cell.i][cell.j] = cell.v;
				}
			}
			else
			{			
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
						ret[i][j] = mb.getValueDenseUnsafe(i, j);
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param mb
	 * @return
	 */
	public static boolean [] convertToBooleanVector(MatrixBlock mb)
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		boolean[] ret = new boolean[rows*cols]; //false-initialized 
		
		if( mb.getNonZeros() > 0 )
		{
			if( mb.isInSparseFormat() )
			{
				Iterator<IJV> iter = mb.getSparseBlockIterator();
				while( iter.hasNext() )
				{
					IJV cell = iter.next();
					ret[cell.i*cols+cell.j] = (cell.v != 0.0);
				}
			}
			else
			{
				for( int i=0, cix=0; i<rows; i++ )
					for( int j=0; j<cols; j++, cix++)
						ret[cix] = (mb.getValueDenseUnsafe(i, j) != 0.0);
			}
		}
		
		return ret;
	}
	
	
	/**
	 * 
	 * @param mb
	 * @return
	 */
	public static int[] convertToIntVector( MatrixBlock mb)
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		int[] ret = new int[rows*cols]; //0-initialized
		
		
		if( mb.getNonZeros() > 0 )
		{
			if( mb.isInSparseFormat() )
			{
				Iterator<IJV> iter = mb.getSparseBlockIterator();
				while( iter.hasNext() )
				{
					IJV cell = iter.next();
					ret[cell.i*cols+cell.j] = (int)cell.v;
				}
			}
			else
			{
				//memcopy row major representation if at least 1 non-zero
				for( int i=0, cix=0; i<rows; i++ )
					for( int j=0; j<cols; j++, cix++ )
						ret[cix] = (int)(mb.getValueDenseUnsafe(i, j));
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param mb
	 * @return
	 */
	public static double[] convertToDoubleVector( MatrixBlock mb )
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		double[] ret = new double[rows*cols]; //0-initialized 
		
		if( mb.getNonZeros() > 0 )
		{
			if( mb.isInSparseFormat() )
			{
				Iterator<IJV> iter = mb.getSparseBlockIterator();
				while( iter.hasNext() )
				{
					IJV cell = iter.next();
					ret[cell.i*cols+cell.j] = cell.v;
				}
			}
			else
			{
				//memcopy row major representation if at least 1 non-zero
				System.arraycopy(mb.getDenseBlock(), 0, ret, 0, rows*cols);
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param mb
	 * @return
	 */
	public static List<Double> convertToDoubleList( MatrixBlock mb )
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		long nnz = mb.getNonZeros();
		ArrayList<Double> ret = new ArrayList<Double>();
		
		if( mb.isInSparseFormat() )
		{
			Iterator<IJV> iter = mb.getSparseBlockIterator();
			while( iter.hasNext() )
			{
				IJV cell = iter.next();
				ret.add( cell.v );
			}
			for( long i=nnz; i<(long)rows*cols; i++ )
				ret.add( 0d ); //add remaining values
		}
		else
		{
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					ret.add( mb.getValueDenseUnsafe(i, j) );
		}
				
		return ret;
	}
	
	/**
	 * Creates a dense Matrix Block and copies the given double matrix into it.
	 * 
	 * @param data
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock convertToMatrixBlock( double[][] data ) 
		throws DMLRuntimeException
	{
		int rows = data.length;
		int cols = (rows > 0)? data[0].length : 0;
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		try
		{ 
			//copy data to mb (can be used because we create a dense matrix)
			mb.init( data, rows, cols );
		} 
		catch (Exception e){} //can never happen
		
		//check and convert internal representation
		mb.examSparsity();
		
		return mb;
	}

	/**
	 * Creates a dense Matrix Block and copies the given double vector into it.
	 * 
	 * @param data
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock convertToMatrixBlock( double[] data, boolean columnVector ) 
		throws DMLRuntimeException
	{
		int rows = columnVector ? data.length : 1;
		int cols = columnVector ? 1 : data.length;
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		
		try
		{ 
			//copy data to mb (can be used because we create a dense matrix)
			mb.init( data, rows, cols );
		} 
		catch (Exception e){} //can never happen
		
		//check and convert internal representation
		mb.examSparsity();
		
		return mb;
	}

	/**
	 * 
	 * @param map
	 * @return
	 */
	public static MatrixBlock convertToMatrixBlock( HashMap<MatrixIndexes,Double> map )
	{
		// compute dimensions from the map
		long nrows=0, ncols=0;
		for (MatrixIndexes index : map.keySet()) {
			nrows = Math.max( nrows, index.getRowIndex() );
			ncols = Math.max( ncols, index.getColumnIndex() );
		}
		
		// convert to matrix block
		return convertToMatrixBlock(map, (int)nrows, (int)ncols);
	}
	
	/**
	 * NOTE: this method also ensures the specified matrix dimensions
	 * 
	 * @param map
	 * @return
	 */
	public static MatrixBlock convertToMatrixBlock( HashMap<MatrixIndexes,Double> map, int rlen, int clen )
	{
		int nnz = map.size();
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz); 		
		MatrixBlock mb = new MatrixBlock(rlen, clen, sparse, nnz);
		
		// copy map values into new block
		if( sparse ) //SPARSE <- cells
		{
			//append cells to sparse target (prevent shifting)
			for( Entry<MatrixIndexes,Double> e : map.entrySet() ) 
			{
				MatrixIndexes index = e.getKey();
				double value = e.getValue();
				int rix = (int)index.getRowIndex();
				int cix = (int)index.getColumnIndex();
				if( value != 0 && rix<=rlen && cix<=clen )
					mb.appendValue( rix-1, cix-1, value );
			}
			
			//sort sparse target representation
			mb.sortSparseRows();
		}
		else  //DENSE <- cells
		{
			//directly insert cells into dense target 
			for( Entry<MatrixIndexes,Double> e : map.entrySet() ) 
			{
				MatrixIndexes index = e.getKey();
				double value = e.getValue();
				int rix = (int)index.getRowIndex();
				int cix = (int)index.getColumnIndex();
				if( value != 0 && rix<=rlen && cix<=clen )
					mb.quickSetValue( rix-1, cix-1, value );
			}
		}
		
		return mb;
	}

	/**
	 * 
	 * @param map
	 * @return
	 */
	public static MatrixBlock convertToMatrixBlock( CTableMap map )
	{
		// compute dimensions from the map
		int nrows = (int)map.getMaxRow();
		int ncols = (int)map.getMaxColumn();
		
		// convert to matrix block
		return convertToMatrixBlock(map, nrows, ncols);
	}
	
	/**
	 * NOTE: this method also ensures the specified matrix dimensions
	 * 
	 * @param map
	 * @return
	 */
	public static MatrixBlock convertToMatrixBlock( CTableMap map, int rlen, int clen )
	{
		return map.toMatrixBlock(rlen, clen);
	}
	
	/**
	 * Converts a frame block with arbitrary schema into a matrix block. 
	 * Since matrix block only supports value type double, we do a best 
	 * effort conversion of non-double types which might result in errors 
	 * for non-numerical data.
	 * 
	 * @param frame
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock convertToMatrixBlock(FrameBlock frame) 
		throws DMLRuntimeException
	{
		MatrixBlock mb = new MatrixBlock(frame.getNumRows(), frame.getNumColumns(), false);
		
		List<ValueType> schema = frame.getSchema();
		for( int i=0; i<frame.getNumRows(); i++ ) 
			for( int j=0; j<frame.getNumColumns(); j++ ) {
				mb.appendValue(i, j, UtilFunctions.objectToDouble(
						schema.get(j), frame.get(i, j)));
			}
		mb.examSparsity();
		
		return mb;
	}
	
	/**
	 * Converts a frame block with arbitrary schema into a two dimensional
	 * string array. 
	 * 
	 * @param frame
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static String[][] convertToStringFrame(FrameBlock frame) 
		throws DMLRuntimeException
	{
		String[][] ret = new String[frame.getNumRows()][];
		Iterator<String[]> iter = frame.getStringRowIterator();		
		for( int i=0; iter.hasNext(); i++ ) {
			//deep copy output rows due to internal reuse
			ret[i] = iter.next().clone();
		}
		
		return ret;
	}
	
	/**
	 * Converts a two dimensions string array into a frame block of 
	 * value type string. If the given array is null or of length 0, 
	 * we return an empty frame block.
	 * 
	 * @param data
	 * @return
	 */
	public static FrameBlock convertToFrameBlock(String[][] data) {
		//check for empty frame block 
		if( data == null || data.length==0 )
			return new FrameBlock();
		
		//create schema and frame block
		List<ValueType> schema = Collections.nCopies(data[0].length, ValueType.STRING);
		return convertToFrameBlock(data, schema);
	}
	
	/**
	 * 
	 * @param data
	 * @param schema
	 * @return
	 */
	public static FrameBlock convertToFrameBlock(String[][] data, List<ValueType> schema) {
		//check for empty frame block 
		if( data == null || data.length==0 )
			return new FrameBlock();
		
		//create frame block
		return new FrameBlock(schema, data);
	}
	
	/**
	 * 
	 * @param data
	 * @param schema
	 * @param colnames
	 * @return
	 */
	public static FrameBlock convertToFrameBlock(String[][] data, List<ValueType> schema, List<String> colnames) {
		//check for empty frame block 
		if( data == null || data.length==0 )
			return new FrameBlock();
		
		//create frame block
		return new FrameBlock(schema, colnames, data);
	}
	
	/**
	 * Converts a matrix block into a frame block of value type double.
	 * 
	 * @param mb
	 * @return
	 */
	public static FrameBlock convertToFrameBlock(MatrixBlock mb) {
		return convertToFrameBlock(mb, ValueType.DOUBLE);
	}
	
	/**
	 * Converts a matrix block into a frame block of a given value type.
	 * 
	 * @param mb
	 * @param vt
	 * @return
	 */
	public static FrameBlock convertToFrameBlock(MatrixBlock mb, ValueType vt) {
		//create schema and frame block
		List<ValueType> schema = Collections.nCopies(mb.getNumColumns(), vt);
		return convertToFrameBlock(mb, schema);
	}
	
	/**
	 * 
	 * @param mb
	 * @param schema
	 * @return
	 */
	public static FrameBlock convertToFrameBlock(MatrixBlock mb, List<ValueType> schema)
	{
		FrameBlock frame = new FrameBlock(schema);
		Object[] row = new Object[mb.getNumColumns()];
		
		if( mb.isInSparseFormat() ) //SPARSE
		{
			SparseBlock sblock = mb.getSparseBlock();			
			for( int i=0; i<mb.getNumRows(); i++ ) {
				Arrays.fill(row, null); //reset
				if( !sblock.isEmpty(i) ) {
					int apos = sblock.pos(i);
					int alen = sblock.size(i);
					int[] aix = sblock.indexes(i);
					double[] aval = sblock.values(i);
					for( int j=apos; j<apos+alen; j++ ) {
						row[aix[j]] = UtilFunctions.doubleToObject(
								schema.get(aix[j]), aval[j]);					
					}
				}
				frame.appendRow(row);
			}
		}
		else //DENSE
		{
			for( int i=0; i<mb.getNumRows(); i++ ) {
				Arrays.fill(row, null); //reset
				for( int j=0; j<mb.getNumColumns(); j++ ) {
					row[j] = UtilFunctions.doubleToObject(
							schema.get(j), 
							mb.quickGetValue(i, j));
				}
				frame.appendRow(row);
			}
		}
		
		return frame;
	}
	
	/**
	 * 
	 * @param mb
	 * @param colwise
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock[] convertToMatrixBlockPartitions( MatrixBlock mb, boolean colwise ) 
		throws DMLRuntimeException
	{
		MatrixBlock[] ret = null;
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		long nnz = mb.getNonZeros();
		boolean sparse = mb.isInSparseFormat();
		double sparsity = ((double)nnz)/(rows*cols);
		
		if( colwise ) //COL PARTITIONS
		{
			//allocate output partitions
			ret = new MatrixBlock[ cols ];
			for( int j=0; j<cols; j++ )
				ret[j] = new MatrixBlock(rows, 1, false);

			//cache-friendly sequential read/append
			if( !mb.isEmptyBlock(false) ) {
				if( sparse ){ //SPARSE
					Iterator<IJV> iter = mb.getSparseBlockIterator();
					while( iter.hasNext() ) {
						IJV cell = iter.next();
						ret[cell.j].appendValue(cell.i, 0, cell.v);
					}
				}
				else { //DENSE
					for( int i=0; i<rows; i++ )
						for( int j=0; j<cols; j++ )
							ret[j].appendValue(i, 0, mb.getValueDenseUnsafe(i, j));
				}
			}
		}
		else //ROW PARTITIONS
		{
			//allocate output partitions
			ret = new MatrixBlock[ rows ];
			for( int i=0; i<rows; i++ )
				ret[i] = new MatrixBlock(1, cols, sparse, (long)(cols*sparsity));

			//cache-friendly sparse/dense row slicing 
			if( !mb.isEmptyBlock(false) ) {
				for( int i=0; i<rows; i++ )
					mb.sliceOperations(i, i, 0, cols-1, ret[i]);
			}			
		}
		
		return ret;
	}
	
	/**
	 * Helper method that converts SystemML matrix variable (<code>varname</code>) into a Array2DRowRealMatrix format,
	 * which is useful in invoking Apache CommonsMath.
	 * 
	 * @param ec
	 * @param varname
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Array2DRowRealMatrix convertToArray2DRowRealMatrix(MatrixObject mo) 
		throws DMLRuntimeException 
	{
		Matrix.ValueType vt = (mo.getValueType() == ValueType.DOUBLE ? Matrix.ValueType.Double : Matrix.ValueType.Integer);
		Matrix mathInput = new Matrix(mo.getFileName(), mo.getNumRows(), mo.getNumColumns(), vt);
		mathInput.setMatrixObject(mo);
		double[][] data = mathInput.getMatrixAsDoubleArray();
		Array2DRowRealMatrix matrixInput = new Array2DRowRealMatrix(data, false);
		
		return matrixInput;
	}
}
