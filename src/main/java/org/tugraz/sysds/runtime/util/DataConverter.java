/*
 * Modifications copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.util;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.data.DenseBlock;
import org.tugraz.sysds.runtime.data.DenseBlockFactory;
import org.tugraz.sysds.runtime.data.DataTensor;
import org.tugraz.sysds.runtime.data.SparseBlock;
import org.tugraz.sysds.runtime.data.BasicTensor;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.instructions.cp.BooleanObject;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.ListObject;
import org.tugraz.sysds.runtime.instructions.cp.ScalarObject;
import org.tugraz.sysds.runtime.io.FileFormatProperties;
import org.tugraz.sysds.runtime.io.MatrixReader;
import org.tugraz.sysds.runtime.io.MatrixReaderFactory;
import org.tugraz.sysds.runtime.io.MatrixWriter;
import org.tugraz.sysds.runtime.io.MatrixWriterFactory;
import org.tugraz.sysds.runtime.io.ReadProperties;
import org.tugraz.sysds.runtime.matrix.data.CTableMap;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.IJV;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.StringTokenizer;


/**
 * This class provides methods to read and write matrix blocks from to HDFS using different data formats.
 * Those functionalities are used especially for CP read/write and exporting in-memory matrices to HDFS
 * (before executing MR jobs).
 * 
 */
public class DataConverter 
{
	private static final String DELIM = " ";
	
	//////////////
	// READING and WRITING of matrix blocks to/from HDFS
	// (textcell, binarycell, binaryblock)
	///////

	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo,  DataCharacteristics dc )
		throws IOException {
		writeMatrixToHDFS(mat, dir, outputinfo, dc, -1, null);
	}

	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo, DataCharacteristics dc, int replication, FileFormatProperties formatProperties)
		throws IOException {
		writeMatrixToHDFS(mat, dir, outputinfo, dc, -1, null, false);
	}
	
	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo, DataCharacteristics dc, int replication, FileFormatProperties formatProperties, boolean diag)
		throws IOException {
		MatrixWriter writer = MatrixWriterFactory.createMatrixWriter( outputinfo, replication, formatProperties );
		writer.writeMatrixToHDFS(mat, dir, dc.getRows(), dc.getCols(), dc.getBlocksize(), dc.getNonZeros(), diag);
	}

	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int blen, boolean localFS) 
		throws IOException
	{	
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.blen = blen;
		prop.localFS = localFS;
		
		return readMatrixFromHDFS(prop);
	}

	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int blen) 
		throws IOException
	{	
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.blen = blen;
		
		return readMatrixFromHDFS(prop);
	}

	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int blen, long expectedNnz) 
		throws IOException
	{
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.blen = blen;
		prop.expectedNnz = expectedNnz;
		
		return readMatrixFromHDFS(prop);
	}

	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int blen, long expectedNnz, boolean localFS) 
		throws IOException
	{
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.blen = blen;
		prop.expectedNnz = expectedNnz;
		prop.localFS = localFS;
		
		return readMatrixFromHDFS(prop);
	}

	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int blen, long expectedNnz, FileFormatProperties formatProperties) 
	throws IOException
	{
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.blen = blen;
		prop.expectedNnz = expectedNnz;
		prop.formatProperties = formatProperties;
		
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
	 *  * best (ordered, or binary block w/ clen&lt;=blen): O(M*N)
	 *  * average (unordered): O(M*N*log(N))
	 *  * worst (descending order per row): O(M * N^2)
	 * 
	 * NOTE: providing an exact estimate of 'expected sparsity' can prevent a full copy of the result
	 * matrix block (required for changing sparse-&gt;dense, or vice versa)
	 * 
	 * @param prop read properties
	 * @return matrix block
	 * @throws IOException if IOException occurs
	 */
	public static MatrixBlock readMatrixFromHDFS(ReadProperties prop) 
		throws IOException
	{	
		//Timing time = new Timing(true);
		
		//core matrix reading 
		MatrixBlock ret = null;
		try {
			MatrixReader reader = MatrixReaderFactory.createMatrixReader(prop);
			ret = reader.readMatrixFromHDFS(prop.path, prop.rlen, prop.clen, prop.blen, prop.expectedNnz);
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
	 * @param mb matrix block
	 * @return 2d double array
	 */
	public static double[][] convertToDoubleMatrix( MatrixBlock mb ) {
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		double[][] ret = new double[rows][cols]; //0-initialized
		
		if( mb.getNonZeros() > 0 ) {
			if( mb.isInSparseFormat() ) {
				Iterator<IJV> iter = mb.getSparseBlockIterator();
				while( iter.hasNext() ) {
					IJV cell = iter.next();
					ret[cell.getI()][cell.getJ()] = cell.getV();
				}
			}
			else {
				double[] a = mb.getDenseBlockValues();
				for( int i=0, ix=0; i<rows; i++ )
					for( int j=0; j<cols; j++, ix++ )
						ret[i][j] = a[ix];
			}
		}
		return ret;
	}

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
				while( iter.hasNext() ) {
					IJV cell = iter.next();
					ret[cell.getI()*cols+cell.getJ()] = (cell.getV() != 0.0);
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

	public static int[] convertToIntVector( MatrixBlock mb) {
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		int[] ret = new int[rows*cols]; //0-initialized
		if( mb.isEmptyBlock(false) ) 
			return ret;
		if( mb.isInSparseFormat() ) {
			Iterator<IJV> iter = mb.getSparseBlockIterator();
			while( iter.hasNext() ) {
				IJV cell = iter.next();
				ret[cell.getI()*cols+cell.getJ()] = (int)cell.getV();
			}
		}
		else {
			//memcopy row major representation if at least 1 non-zero
			for( int i=0, cix=0; i<rows; i++ )
				for( int j=0; j<cols; j++, cix++ )
					ret[cix] = (int)(mb.getValueDenseUnsafe(i, j));
		}
		return ret;
	}
	
	public static long[] convertToLongVector( MatrixBlock mb) {
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		long[] ret = new long[rows*cols]; //0-initialized
		if( mb.isEmptyBlock(false) ) 
			return ret;
		if( mb.isInSparseFormat() ) {
			Iterator<IJV> iter = mb.getSparseBlockIterator();
			while( iter.hasNext() ) {
				IJV cell = iter.next();
				ret[cell.getI()*cols+cell.getJ()] = (int)cell.getV();
			}
		}
		else {
			//memcopy row major representation if at least 1 non-zero
			for( int i=0, cix=0; i<rows; i++ )
				for( int j=0; j<cols; j++, cix++ )
					ret[cix] = (int)(mb.getValueDenseUnsafe(i, j));
		}
		return ret;
	}
	
	public static DenseBlock convertToDenseBlock(MatrixBlock mb) {
		return convertToDenseBlock(mb, true);
	}
	
	public static DenseBlock convertToDenseBlock(MatrixBlock mb, boolean deep) {
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		DenseBlock ret = (!mb.isInSparseFormat() && mb.isAllocated() && !deep) ? 
			mb.getDenseBlock() : DenseBlockFactory.createDenseBlock(rows, cols); //0-initialized
		
		if( !mb.isEmptyBlock(false) ) {
			if( mb.isInSparseFormat() ) {
				Iterator<IJV> iter = mb.getSparseBlockIterator();
				while( iter.hasNext() ) {
					IJV cell = iter.next();
					ret.set(cell.getI(), cell.getJ(), cell.getV());
				}
			}
			else if( deep ) {
				ret.set(mb.getDenseBlock());
			}
		}
		
		return ret;
	}

	public static double[] convertToDoubleVector(MatrixBlock mb) {
		return convertToDoubleVector(mb, true);
	}
	
	public static double[] convertToDoubleVector( MatrixBlock mb, boolean deep ) {
		return convertToDoubleVector(mb, deep, false);
	}
	
	public static double[] convertToDoubleVector( MatrixBlock mb, boolean deep, boolean allowNull )
	{
		if( mb.isEmpty() && allowNull )
			return null;
		
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		double[] ret = (!mb.isInSparseFormat() && mb.isAllocated() && !deep) ? 
			mb.getDenseBlockValues() : new double[rows*cols]; //0-initialized
		
		if( !mb.isEmptyBlock(false) ) {
			if( mb.isInSparseFormat() ) {
				Iterator<IJV> iter = mb.getSparseBlockIterator();
				while( iter.hasNext() ) {
					IJV cell = iter.next();
					ret[cell.getI()*cols+cell.getJ()] = cell.getV();
				}
			}
			else if( deep ) {
				//memcopy row major representation if at least 1 non-zero
				System.arraycopy(mb.getDenseBlockValues(), 0, ret, 0, rows*cols);
			}
		}
		
		return ret;
	}

	public static List<Double> convertToDoubleList( MatrixBlock mb )
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		long nnz = mb.getNonZeros();
		ArrayList<Double> ret = new ArrayList<>();
		
		if( mb.isInSparseFormat() )
		{
			Iterator<IJV> iter = mb.getSparseBlockIterator();
			while( iter.hasNext() ) {
				IJV cell = iter.next();
				ret.add( cell.getV() );
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
	 * @param data 2d double array
	 * @return matrix block
	 */
	public static MatrixBlock convertToMatrixBlock( double[][] data ) {
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
	 * @param data double array
	 * @param columnVector if true, create matrix with single column. if false, create matrix with single row
	 * @return matrix block
	 */
	public static MatrixBlock convertToMatrixBlock( double[] data, boolean columnVector ) {
		int rows = columnVector ? data.length : 1;
		int cols = columnVector ? 1 : data.length;
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		//copy data to mb (can be used because we create a dense matrix)
		mb.init( data, rows, cols );
		mb.examSparsity();
		return mb;
	}

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
	 * @param map map of matrix index keys and double values
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @return matrix block
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
	 * @param map ?
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @return matrix block
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
	 * @param frame frame block
	 * @return matrix block
	 */
	public static MatrixBlock convertToMatrixBlock(FrameBlock frame) 
	{
		int m = frame.getNumRows();
		int n = frame.getNumColumns();
		MatrixBlock mb = new MatrixBlock(m, n, false);
		mb.allocateDenseBlock();
		
		ValueType[] schema = frame.getSchema();
		int dFreq = UtilFunctions.frequency(schema, ValueType.FP64);
		
		if( dFreq == schema.length ) {
			// special case double schema (without cell-object creation, 
			// cache-friendly row-column copy)
			double[][] a = new double[n][];
			double[] c = mb.getDenseBlockValues();
			for( int j=0; j<n; j++ )
				a[j] = (double[])frame.getColumnData(j);
			int blocksizeIJ = 16; //blocks of a+overhead/c in L1 cache
			for( int bi=0; bi<m; bi+=blocksizeIJ )
				for( int bj=0; bj<n; bj+=blocksizeIJ ) {
					int bimin = Math.min(bi+blocksizeIJ, m);
					int bjmin = Math.min(bj+blocksizeIJ, n);
					for( int i=bi, aix=bi*n; i<bimin; i++, aix+=n )
						for( int j=bj; j<bjmin; j++ )
							c[aix+j] = a[j][i];
				}
		}
		else { 
			//general case
			for( int i=0; i<frame.getNumRows(); i++ ) 
				for( int j=0; j<frame.getNumColumns(); j++ ) {
					mb.appendValue(i, j, UtilFunctions.objectToDouble(
							schema[j], frame.get(i, j)));
				}
		}
		
		//post-processing
		mb.examSparsity();
		
		return mb;
	}
	
	/**
	 * Converts a frame block with arbitrary schema into a two dimensional
	 * string array. 
	 * 
	 * @param frame frame block
	 * @return 2d string array
	 */
	public static String[][] convertToStringFrame(FrameBlock frame) 
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
	 * @param data 2d string array
	 * @return frame block
	 */
	public static FrameBlock convertToFrameBlock(String[][] data) {
		//check for empty frame block 
		if( data == null || data.length==0 )
			return new FrameBlock();
		
		//create schema and frame block
		ValueType[] schema = UtilFunctions.nCopies(data[0].length, ValueType.STRING);
		return convertToFrameBlock(data, schema);
	}

	public static FrameBlock convertToFrameBlock(String[][] data, ValueType[] schema) {
		//check for empty frame block 
		if( data == null || data.length==0 )
			return new FrameBlock();
		
		//create frame block
		return new FrameBlock(schema, data);
	}

	public static FrameBlock convertToFrameBlock(String[][] data, ValueType[] schema, String[] colnames) {
		//check for empty frame block 
		if( data == null || data.length==0 )
			return new FrameBlock();
		
		//create frame block
		return new FrameBlock(schema, colnames, data);
	}
	
	/**
	 * Converts a matrix block into a frame block of value type double.
	 * 
	 * @param mb matrix block
	 * @return frame block of type double
	 */
	public static FrameBlock convertToFrameBlock(MatrixBlock mb) {
		return convertToFrameBlock(mb, ValueType.FP64);
	}
	
	/**
	 * Converts a matrix block into a frame block of a given value type.
	 * 
	 * @param mb matrix block
	 * @param vt value type
	 * @return frame block
	 */
	public static FrameBlock convertToFrameBlock(MatrixBlock mb, ValueType vt) {
		//create schema and frame block
		ValueType[] schema = UtilFunctions.nCopies(mb.getNumColumns(), vt);
		return convertToFrameBlock(mb, schema);
	}

	public static FrameBlock convertToFrameBlock(MatrixBlock mb, ValueType[] schema)
	{
		FrameBlock frame = new FrameBlock(schema);
		Object[] row = new Object[mb.getNumColumns()];
		
		if( mb.isInSparseFormat() ) //SPARSE
		{
			SparseBlock sblock = mb.getSparseBlock();
			for( int i=0; i<mb.getNumRows(); i++ ) {
				Arrays.fill(row, null); //reset
				if( sblock != null && !sblock.isEmpty(i) ) {
					int apos = sblock.pos(i);
					int alen = sblock.size(i);
					int[] aix = sblock.indexes(i);
					double[] aval = sblock.values(i);
					for( int j=apos; j<apos+alen; j++ ) {
						row[aix[j]] = UtilFunctions.doubleToObject(
								schema[aix[j]], aval[j]);
					}
				}
				frame.appendRow(row);
			}
		}
		else //DENSE
		{
			int dFreq = UtilFunctions.frequency(schema, ValueType.FP64);
			
			if( schema.length==1 && dFreq==1 && mb.isAllocated() ) {
				// special case double schema and single columns which
				// allows for a shallow copy since the physical representation
				// of row-major matrix and column-major frame match exactly
				frame.reset();
				frame.appendColumns(new double[][]{mb.getDenseBlockValues()});
			}
			else if( dFreq == schema.length ) {
				// special case double schema (without cell-object creation, 
				// col pre-allocation, and cache-friendly row-column copy)
				int m = mb.getNumRows();
				int n = mb.getNumColumns();
				double[] a = mb.getDenseBlockValues();
				double[][] c = new double[n][m];
				int blocksizeIJ = 16; //blocks of a/c+overhead in L1 cache
				if( !mb.isEmptyBlock(false) )
					for( int bi=0; bi<m; bi+=blocksizeIJ )
						for( int bj=0; bj<n; bj+=blocksizeIJ ) {
							int bimin = Math.min(bi+blocksizeIJ, m);
							int bjmin = Math.min(bj+blocksizeIJ, n);
							for( int i=bi, aix=bi*n; i<bimin; i++, aix+=n )
								for( int j=bj; j<bjmin; j++ )
									c[j][i] = a[aix+j];
						}
				frame.reset();
				frame.appendColumns(c);
			}
			else { 
				// general case
				for( int i=0; i<mb.getNumRows(); i++ ) {
					for( int j=0; j<mb.getNumColumns(); j++ ) {
							row[j] = UtilFunctions.doubleToObject(
									schema[j], mb.quickGetValue(i, j));
					}
					frame.appendRow(row);
				}
			}
		}
		
		return frame;
	}
	
	public static TensorBlock convertToTensorBlock(MatrixBlock mb, ValueType vt, boolean toBasicTensor) {
		TensorBlock ret;
		if (toBasicTensor) {
			BasicTensor bt = new BasicTensor(vt, new int[] {mb.getNumRows(), mb.getNumColumns()});
			bt.allocateDenseBlock(true);
			ret = bt;
		}
		else {
			DataTensor dt = new DataTensor(vt, new int[] {mb.getNumRows(), mb.getNumColumns()});
			dt.allocateBlock();
			ret = dt;
		}
		if( mb.getNonZeros() > 0 ) {
			if( mb.isInSparseFormat() ) {
				Iterator<IJV> iter = mb.getSparseBlockIterator();
				while( iter.hasNext() ) {
					IJV cell = iter.next();
					ret.set(cell.getI(), cell.getJ(), cell.getV());
				}
			}
			else {
				double[] a = mb.getDenseBlockValues();
				for( int i=0, ix=0; i<mb.getNumRows(); i++ )
					for( int j=0; j<mb.getNumColumns(); j++, ix++ )
						ret.set(i, j, a[ix]);
			}
		}
		return ret;
	}

	public static MatrixBlock[] convertToMatrixBlockPartitions( MatrixBlock mb, boolean colwise )
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
						ret[cell.getJ()].appendValue(cell.getI(), 0, cell.getV());
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
					mb.slice(i, i, 0, cols-1, ret[i]);
			}
		}
		
		return ret;
	}
	
	/**
	 * Helper method that converts SystemDS matrix variable (<code>varname</code>) into a Array2DRowRealMatrix format,
	 * which is useful in invoking Apache CommonsMath.
	 * 
	 * @param mb matrix object
	 * @return matrix as a commons-math3 Array2DRowRealMatrix
	 */
	public static Array2DRowRealMatrix convertToArray2DRowRealMatrix(MatrixBlock mb) {
		double[][] data = DataConverter.convertToDoubleMatrix(mb);
		return new Array2DRowRealMatrix(data, false);
	}
	
	public static BlockRealMatrix convertToBlockRealMatrix(MatrixBlock mb) {
		BlockRealMatrix ret = new BlockRealMatrix(mb.getNumRows(), mb.getNumColumns());
		if( mb.getNonZeros() > 0 ) {
			if( mb.isInSparseFormat() ) {
				Iterator<IJV> iter = mb.getSparseBlockIterator();
				while( iter.hasNext() ) {
					IJV cell = iter.next();
					ret.setEntry(cell.getI(), cell.getJ(), cell.getV());
				}
			}
			else {
				double[] a = mb.getDenseBlockValues();
				for( int i=0, ix=0; i<mb.getNumRows(); i++ )
					for( int j=0; j<mb.getNumColumns(); j++, ix++ )
						ret.setEntry(i, j, a[ix]);
			}
		}
		return ret;
	}
	
	public static MatrixBlock convertToMatrixBlock(RealMatrix rm) {
		MatrixBlock ret = new MatrixBlock(rm.getRowDimension(),
			rm.getColumnDimension(), false).allocateDenseBlock();
		for(int i=0; i<ret.getNumRows(); i++)
			for(int j=0; j<ret.getNumColumns(); j++)
				ret.quickSetValue(i, j, rm.getEntry(i, j));
		ret.examSparsity();
		return ret;
	}

	public static void copyToDoubleVector( MatrixBlock mb, double[] dest, int destPos )
	{
		if( mb.isEmptyBlock(false) )
			return; //quick path
			
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		
		if( mb.isInSparseFormat() ) {
			Iterator<IJV> iter = mb.getSparseBlockIterator();
			while( iter.hasNext() ) {
				IJV cell = iter.next();
				dest[destPos+cell.getI()*cols+cell.getJ()] = cell.getV();
			}
		}
		else {
			//memcopy row major representation if at least 1 non-zero
			System.arraycopy(mb.getDenseBlockValues(), 0, dest, destPos, rows*cols);
		}
	}
	
	/**
	 * Convenience method to print NaN & Infinity compliant with how as.scalar prints them.
	 * {@link DecimalFormat} prints NaN as \uFFFD and Infinity as \u221E
	 * http://docs.oracle.com/javase/6/docs/api/java/text/DecimalFormat.html
	 * @param df	The {@link DecimalFormat} instance, constructed with the appropriate options
	 * @param value	The double value to print
	 * @return	a string formatted with the {@link DecimalFormat} instance or "NaN" or "Infinity" or "-Infinity"
	 */
	private static String dfFormat(DecimalFormat df, double value) {
		if (Double.isNaN(value) || Double.isInfinite(value)){
			return Double.toString(value);
		} else {
			return df.format(value);
		}
	}

	public static String toString(MatrixBlock mb) {
		return toString(mb, false, " ", "\n", mb.getNumRows(), mb.getNumColumns(), 3);
	}
	
	/**
	 * Returns a string representation of a matrix
	 * @param mb matrix block
	 * @param sparse if true, string will contain a table with row index, col index, value (where value != 0.0)
	 * 				 otherwise it will be a rectangular string with all values of the matrix block
	 * @param separator Separator string between each element in a row, or between the columns in sparse format
	 * @param lineseparator Separator string between each row
	 * @param rowsToPrint maximum number of rows to print, -1 for all
	 * @param colsToPrint maximum number of columns to print, -1 for all
	 * @param decimal number of decimal places to print, -1 for default
	 * @return matrix as a string
	 */
	public static String toString(MatrixBlock mb, boolean sparse, String separator, String lineseparator, int rowsToPrint, int colsToPrint, int decimal){
		StringBuffer sb = new StringBuffer();
		
		// Setup number of rows and columns to print
		int rlen = mb.getNumRows();
		int clen = mb.getNumColumns();
		int rowLength = rlen;
		int colLength = clen;
		if (rowsToPrint >= 0)
			rowLength = rowsToPrint < rlen ? rowsToPrint : rlen;
		if (colsToPrint >= 0)
			colLength = colsToPrint < clen ? colsToPrint : clen;
		
		DecimalFormat df = new DecimalFormat();
		df.setGroupingUsed(false);
		if (decimal >= 0){
			df.setMinimumFractionDigits(decimal);
		}
		
		if (sparse){ // Sparse Print Format
			if (mb.isInSparseFormat()){	// Block is in sparse format
				Iterator<IJV> sbi = mb.getSparseBlockIterator();
				while (sbi.hasNext()){
					IJV ijv = sbi.next();
					int row = ijv.getI();
					int col = ijv.getJ();
					double value = ijv.getV();
					if (row < rowLength && col < colLength) {
						// Print (row+1) and (col+1) since for a DML user, everything is 1-indexed
						sb.append(row+1).append(separator).append(col+1).append(separator);
						sb.append(dfFormat(df, value)).append(lineseparator);
					}
				}
			} else {	// Block is in dense format
				for (int i=0; i<rowLength; i++){
					for (int j=0; j<colLength; j++){
						double value = mb.getValue(i, j);
						if (value != 0.0){
							sb.append(i+1).append(separator).append(j+1).append(separator);
							sb.append(dfFormat(df, value)).append(lineseparator);
						}
					}
				}
			}
		}
		else {	// Dense Print Format
			for (int i=0; i<rowLength; i++){
				for (int j=0; j<colLength-1; j++){
					Double value = mb.quickGetValue(i, j);
					if (value.equals(-0.0d))
						value = 0.0;
					sb.append(dfFormat(df, value));
					sb.append(separator);
				}
				Double value = mb.quickGetValue(i, colLength-1);
				if (value.equals(-0.0d))
					value = 0.0;
				sb.append(dfFormat(df, value));	// Do not put separator after last element
				sb.append(lineseparator);
			}
		}
		
		return sb.toString();
	}

	public static String toString(BasicTensor mb) {
		return toString(mb, false, " ", "\n", "[", "]", mb.getNumRows(), mb.getNumColumns(), 3);
	}

	/**
	 * Returns a string representation of a tensor
	 * @param tb tensor block
	 * @param sparse if true, string will contain a table with row index, col index, value (where value != 0.0)
	 * 				 otherwise it will be a rectangular string with all values of the tensor block
	 * @param separator Separator string between each element in a row, or between the columns in sparse format
	 * @param lineseparator Separator string between each row
	 * @param leftBorder Characters placed at the start of a new dimension level
	 * @param rightBorder Characters placed at the end of a new dimension level
	 * @param rowsToPrint maximum number of rows to print, -1 for all
	 * @param colsToPrint maximum number of columns to print, -1 for all
	 * @param decimal number of decimal places to print, -1 for default
	 * @return tensor as a string
	 */
	public static String toString(BasicTensor tb, boolean sparse, String separator, String lineseparator,
	                              String leftBorder, String rightBorder, int rowsToPrint, int colsToPrint, int decimal){
		StringBuilder sb = new StringBuilder();

		// Setup number of rows and columns to print
		int rlen = tb.getNumRows();
		int clen = tb.getNumColumns();
		int rowLength = rlen;
		int colLength = clen;
		if (rowsToPrint >= 0)
			rowLength = Math.min(rowsToPrint, rlen);
		if (colsToPrint >= 0)
			colLength = Math.min(colsToPrint, clen);

		DecimalFormat df = new DecimalFormat();
		df.setGroupingUsed(false);
		if (decimal >= 0){
			df.setMinimumFractionDigits(decimal);
		}

		if (sparse){ // Sparse Print Format
			// TODO use sparse iterator for sparse block
			int[] ix = new int[tb.getNumDims()];
			for (int i = 0; i < tb.getLength(); i++) {
				String str = tb.get(ix).toString();
				if (str != null && !str.isEmpty() && Double.parseDouble(str) != 0) {
					for (int item : ix) {
						sb.append(item).append(separator);
					}
					concatenateTensorValue(tb, sb, df, ix);
					sb.append(lineseparator);
				}
				tb.getNextIndexes(ix);
				if (ix[0] >= rowLength) {
					break;
				}
			}
		}
		else {	// Dense Print Format
			int[] ix = new int[tb.getNumDims()];
			sb.append(StringUtils.repeat(leftBorder, ix.length));
			for (int i = 0; i < tb.getLength(); i++) {
				concatenateTensorValue(tb, sb, df, ix);
				int j = ix.length - 1;
				ix[j]++;
				//calculating next index
				int borderCount = 0;
				while (ix[j] == tb.getDim(j) || ix[1] >= colLength) {
					// we either reached the dimension limit or the colLength if j == 1
					// so we add border (because of the completely iterated dimension) and increment the next
					// dimension while setting the current one to 0
					if (ix[j] != tb.getDim(j))
						sb.append("...");
					sb.append(rightBorder);
					borderCount++;
					ix[j] = 0;
					j--;
					if (j < 0) {
						break;
					}
					ix[j]++;
				}
				if (ix[0] >= rowLength) {
					// If we have a limit on rows end here
					sb.append("...").append(rightBorder).append(lineseparator);
					break;
				}
				if (j < 0) {
					// we are at the end, no offset
					sb.append(lineseparator);
					break;
				}
				if (borderCount == 0) {
					sb.append(separator);
				} else {
					// Offset so dimensions are aligned
					sb.append(lineseparator);
					sb.append(StringUtils.repeat(" ", (ix.length - borderCount) * leftBorder.length()));
					sb.append(StringUtils.repeat(leftBorder, borderCount));
				}
			}
		}
		return sb.toString();
	}

	/**
	 * Concatenates a single tensor value to the `StringBuilder` by converting it to the correct format.
	 *
	 * @param tb the BasicTensor
	 * @param sb the StringBuilder to use
	 * @param df DecimalFormat with the correct settings for double or float values
	 * @param ix the index of the TensorBlock value
	 */
	private static void concatenateTensorValue(BasicTensor tb, StringBuilder sb, DecimalFormat df, int[] ix) {
		switch (tb.getValueType()) {
			case FP32:
				Float valuef = (Float) tb.get(ix);
				if (valuef.equals(-0.0f))
					valuef = 0.0f;
				sb.append(dfFormat(df, valuef));
				break;
			case FP64:
				Double value = (Double) tb.get(ix);
				if (value.equals(-0.0d))
					value = 0.0;
				sb.append(dfFormat(df, value));
				break;
			case INT32:
			case INT64:
				sb.append(tb.get(ix));
				break;
			case BOOLEAN:
				sb.append(((Boolean) tb.get(ix)).toString().toUpperCase());
				break;
			case STRING:
			case UNKNOWN:
				sb.append("\"").append(tb.get(ix)).append("\"");
				break;
		}
	}

	public static String toString(FrameBlock fb) {
		return toString(fb, false, " ", "\n", fb.getNumRows(), fb.getNumColumns(), 3);
	}

	public static String toString(FrameBlock fb, boolean sparse, String separator, String lineseparator, int rowsToPrint, int colsToPrint, int decimal)
	{
		StringBuffer sb = new StringBuffer();
		
		// Setup number of rows and columns to print
		int rlen = fb.getNumRows();
		int clen = fb.getNumColumns();
		int rowLength = rlen;
		int colLength = clen;
		if (rowsToPrint >= 0)
			rowLength = rowsToPrint < rlen ? rowsToPrint : rlen;
		if (colsToPrint >= 0)
			colLength = colsToPrint < clen ? colsToPrint : clen;
		
		//print frame header
		sb.append("# FRAME: ");
		sb.append("nrow = " + fb.getNumRows() + ", ");
		sb.append("ncol = " + fb.getNumColumns() + lineseparator);
		
		//print column names
		sb.append("#"); sb.append(separator);
		for( int j=0; j<colLength; j++ ) {
			sb.append(fb.getColumnNames()[j]);
			if( j != colLength-1 )
				sb.append(separator);
		}
		sb.append(lineseparator);
		
		//print schema
		sb.append("#"); sb.append(separator);
		for( int j=0; j<colLength; j++ ) {
			sb.append(fb.getSchema()[j]);
			if( j != colLength-1 )
				sb.append(separator);
		}
		sb.append(lineseparator);
		
		//print data
		DecimalFormat df = new DecimalFormat();
		df.setGroupingUsed(false);
		if (decimal >= 0)
			df.setMinimumFractionDigits(decimal);
		
		Iterator<Object[]> iter = fb.getObjectRowIterator(0, rowLength);
		while( iter.hasNext() ) {
			Object[] row = iter.next();
			for( int j=0; j<colLength; j++ ) {
				if( row[j]==null )
					sb.append(String.valueOf(row[j]));
				else if( fb.getSchema()[j] == ValueType.FP64 )
					sb.append(dfFormat(df, (Double)row[j]));
				else if( fb.getSchema()[j] == ValueType.BOOLEAN )
					sb.append(new BooleanObject((Boolean)row[j])
						.getLanguageSpecificStringValue());
				else
					sb.append(row[j]);
				if( j != colLength-1 )
					sb.append(separator);
			}
			sb.append(lineseparator);
		}
		
		return sb.toString();
	}

	public static int[] getTensorDimensions(ExecutionContext ec, CPOperand dims) {
		int[] tDims;
		switch (dims.getDataType()) {
			case SCALAR: {
				// Dimensions given as string
				if (dims.getValueType() != Types.ValueType.STRING) {
					throw new DMLRuntimeException("Dimensions have to be passed as list, string, matrix or tensor.");
				}
				String dimensionString = ec.getScalarInput(dims.getName(), Types.ValueType.STRING, dims.isLiteral())
						.getStringValue();
				StringTokenizer dimensions = new StringTokenizer(dimensionString, DELIM);
				tDims = new int[dimensions.countTokens()];
				Arrays.setAll(tDims, (i) -> Integer.parseInt(dimensions.nextToken()));
			}
			break;
			case MATRIX: {
				// Dimensions given as vector
				MatrixBlock in = ec.getMatrixInput(dims.getName());
				boolean colVec = false;
				if (in.getNumRows() == 1) {
					colVec = true;
				} else if (!(in.getNumColumns() == 1)) {
					throw new DMLRuntimeException("Dimensions matrix has to be a vector.");
				}
				tDims = new int[(int) in.getLength()];
				for (int i = 0; i < in.getLength(); i++) {
					tDims[i] = UtilFunctions.toInt(in.getValue(colVec ? 0 : i, colVec ? i : 0));
				}
				ec.releaseMatrixInput(dims.getName());
			}
			break;
			case TENSOR: {
				// Dimensions given as vector
				TensorBlock in = ec.getTensorInput(dims.getName());
				boolean colVec = false;
				if (!in.isVector()) {
					throw new DMLRuntimeException("Dimensions tensor has to be a vector.");
				} else if (in.getNumRows() == 1) {
					colVec = true;
				}
				tDims = new int[(int) in.getLength()];
				for (int i = 0; i < in.getLength(); i++) {
					tDims[i] = UtilFunctions.toInt(in.get(new int[]{colVec ? 0 : i, colVec ? i : 0}));
				}
				ec.releaseTensorInput(dims.getName());
			}
			break;
			case LIST: {
				// Dimensions given as List
				ListObject list = ec.getListObject(dims.getName());
				tDims = new int[list.getLength()];
				List<Data> dimsData = list.getData();
				for (int i = 0; i < tDims.length; i++) {
					if (dimsData.get(i) instanceof ScalarObject) {
						// TODO warning if double value is cast to long?
						tDims[i] = (int) ((ScalarObject) dimsData.get(i)).getLongValue();
					} else {
						throw new DMLRuntimeException("Dims parameter for does not support lists with non scalar values.");
					}
				}
			}
			break;
			default:
				throw new DMLRuntimeException("Dimensions have to be passed as list, string, matrix or tensor.");
		}
		return tDims;
	}

	public static double[] toDouble(float[] data) {
		double[] ret = new double[data.length];
		for(int i=0; i<data.length; i++)
			ret[i] = data[i];
		return ret;
	}
	
	public static double[] toDouble(long[] data) {
		double[] ret = new double[data.length];
		for(int i=0; i<data.length; i++)
			ret[i] = data[i];
		return ret;
	}
	
	public static double[] toDouble(int[] data) {
		double[] ret = new double[data.length];
		for(int i=0; i<data.length; i++)
			ret[i] = data[i];
		return ret;
	}
	
	public static double[] toDouble(BitSet data, int len) {
		double[] ret = new double[len];
		for(int i=0; i<len; i++)
			ret[i] = data.get(i) ? 1 : 0;
		return ret;
	}

	public static double[] toDouble(String[] data) {
		double[] ret = new double[data.length];
		for(int i=0; i<data.length; i++)
			ret[i] = Double.parseDouble(data[i]);
		return ret;
	}

	public static float[] toFloat(double[] data) {
		float[] ret = new float[data.length];
		for( int i=0; i<data.length; i++ )
			ret[i] = (float)data[i];
		return ret;
	}

	public static int[] toInt(double[] data) {
		int[] ret = new int[data.length];
		for( int i=0; i<data.length; i++ )
			ret[i] = UtilFunctions.toInt(data[i]);
		return ret;
	}

	public static long[] toLong(double[] data) {
		long[] ret = new long[data.length];
		for( int i=0; i<data.length; i++ )
			ret[i] = UtilFunctions.toLong(data[i]);
		return ret;
	}

	public static BitSet toBitSet(double[] data) {
		BitSet ret = new BitSet(data.length);
		for( int i=0; i<data.length; i++ )
			ret.set(i, data[i] != 0);
		return ret;
	}

	public static String[] toString(double[] data) {
		String[] ret = new String[data.length];
		for( int i=0; i<data.length; i++ )
			ret[i] = String.valueOf(data[i]);
		return ret;
	}
}
