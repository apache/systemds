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

package org.apache.sysml.runtime.compress;

import java.util.Arrays;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysml.runtime.functionobjects.KahanFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;


/**
 * Base class for column groups encoded with value dictionary.
 * 
 */
public abstract class ColGroupValue extends ColGroup 
{	
	private static final long serialVersionUID = 3786247536054353658L;
		
	public static boolean LOW_LEVEL_OPT = true;	
	
	//sorting of values by physical length helps by 10-20%, especially for serial, while
	//slight performance decrease for parallel incl multi-threaded, hence not applied for
	//distributed operations (also because compression time + garbage collection increases)
	public static final boolean SORT_VALUES_BY_LENGTH = true; 
		
	
	/** Distinct values associated with individual bitmaps. */
	protected double[] _values; //linearized <numcol vals> <numcol vals>
	
	public ColGroupValue() {
		super((int[]) null, -1);
	}
	
	/**
	 * Stores the headers for the individual bitmaps.
	 * 
	 * @param colIndices
	 *            indices (within the block) of the columns included in this
	 *            column
	 * @param numRows
	 *            total number of rows in the parent block
	 * @param ubm
	 *            Uncompressed bitmap representation of the block
	 */
	public ColGroupValue(int[] colIndices, int numRows, UncompressedBitmap ubm) 
	{
		super(colIndices, numRows);

		// sort values by frequency, if requested 
		if( LOW_LEVEL_OPT && SORT_VALUES_BY_LENGTH 
				&& numRows > BitmapEncoder.BITMAP_BLOCK_SZ ) {
			ubm.sortValuesByFrequency();
		}

		// extract and store distinct values (bitmaps handled by subclasses)
		_values = ubm.getValues();
	}

	/**
	 * Constructor for subclass methods that need to create shallow copies
	 * 
	 * @param colIndices
	 *            raw column index information
	 * @param numRows
	 *            number of rows in the block
	 * @param values
	 *            set of distinct values for the block (associated bitmaps are
	 *            kept in the subclass)
	 */
	protected ColGroupValue(int[] colIndices, int numRows, double[] values) {
		super(colIndices, numRows);
		_values = values;
	}
	
	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		
		// adding the size of values
		size += 8; //array reference
		if (_values != null) {
			size += 32 + _values.length * 8; //values
		}
	
		return size;
	}

	/**
	 * Obtain number of distrinct sets of values associated with the bitmaps in this column group.
	 * 
	 * @return the number of distinct sets of values associated with the bitmaps
	 *         in this column group
	 */
	public int getNumValues() {
		return _values.length / _colIndexes.length;
	}

	public double[] getValues() {
		return _values;
	}
	
	protected int containsAllZeroValue() {
		int numVals = getNumValues();
		int numCols = getNumCols();
		for( int i=0, off=0; i<numVals; i++, off+=numCols ) {
			boolean allZeros = true;
			for( int j=0; j<numCols; j++ )
				allZeros &= (_values[off+j] == 0);
			if( allZeros )
				return i;
		}
		return -1;
	}
	
	protected final double sumValues(int valIx) {
		final int numCols = getNumCols();
		final int valOff = valIx * numCols;
		double val = 0.0;
		for( int i = 0; i < numCols; i++ ) {
			val += _values[valOff+i];
		}
		
		return val;
	}
	
	protected final double sumValues(int valIx, KahanFunction kplus, KahanObject kbuff) {
		final int numCols = getNumCols();
		final int valOff = valIx * numCols;
		kbuff.set(0, 0);
		for( int i = 0; i < numCols; i++ ) {
			kplus.execute2(kbuff, _values[valOff+i]);
		}
		
		return kbuff._sum;
	}
	
	protected final double[] sumAllValues(KahanFunction kplus, KahanObject kbuff) {
		//quick path: sum 
		if( getNumCols()==1 && kplus instanceof KahanPlus )
			return _values; //shallow copy of values
		
		//pre-aggregate value tuple 
		final int numVals = getNumValues();
		double[] ret = new double[numVals];
		for( int k=0; k<numVals; k++ )
			ret[k] = sumValues(k, kplus, kbuff);
		
		return ret;
	}
	
	protected final double sumValues(int valIx, double[] b) {
		final int numCols = getNumCols();
		final int valOff = valIx * numCols;
		double val = 0;
		for( int i = 0; i < numCols; i++ ) {
			val += _values[valOff+i] * b[i];
		}
		
		return val;
	}

	protected final double[] preaggValues(int numVals, double[] b) {
		double[] ret = new double[numVals];
		for( int k = 0; k < numVals; k++ )
			ret[k] = sumValues(k, b);
		
		return ret;
	}
	
	/**
	 * NOTE: Shared across OLE/RLE/DDC because value-only computation. 
	 * 
	 * @param result output matrix block
	 * @param builtin function object
	 * @param zeros indicator if column group contains zero values
	 */
	protected void computeMxx(MatrixBlock result, Builtin builtin, boolean zeros) 
	{
		//init and 0-value handling
		double val = Double.MAX_VALUE * ((builtin.getBuiltinCode()==BuiltinCode.MAX)?-1:1);
		if( zeros )
			val = builtin.execute2(val, 0);
		
		//iterate over all values only
		final int numVals = getNumValues();
		final int numCols = getNumCols();		
		for (int k = 0; k < numVals; k++)
			for( int j=0, valOff = k*numCols; j<numCols; j++ )
				val = builtin.execute2(val, _values[ valOff+j ]);
		
		//compute new partial aggregate
		val = builtin.execute2(val, result.quickGetValue(0, 0));
		result.quickSetValue(0, 0, val);
	}
	
	/**
	 * NOTE: Shared across OLE/RLE/DDC because value-only computation. 
	 * 
	 * @param result output matrix block
	 * @param builtin function object
	 * @param zeros indicator if column group contains zero values
	 */
	protected void computeColMxx(MatrixBlock result, Builtin builtin, boolean zeros)
	{
		final int numVals = getNumValues();
		final int numCols = getNumCols();
		
		//init and 0-value handling
		double[] vals = new double[numCols];
		Arrays.fill(vals, Double.MAX_VALUE * ((builtin.getBuiltinCode()==BuiltinCode.MAX)?-1:1));
		if( zeros ) {
			for( int j = 0; j < numCols; j++ )
				vals[j] = builtin.execute2(vals[j], 0);		
		}
		
		//iterate over all values only
		for (int k = 0; k < numVals; k++) 
			for( int j=0, valOff=k*numCols; j<numCols; j++ )
				vals[j] = builtin.execute2(vals[j], _values[ valOff+j ]);
		
		//copy results to output
		for( int j=0; j<numCols; j++ )
			result.quickSetValue(0, _colIndexes[j], vals[j]);
	}
	
	/**
	 * Method for use by subclasses. Applies a scalar operation to the value
	 * metadata stored in the superclass.
	 * 
	 * @param op
	 *            scalar operation to perform
	 * @return transformed copy of value metadata for this column group
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected double[] applyScalarOp(ScalarOperator op)
		throws DMLRuntimeException 
	{
		//scan over linearized values
		double[] ret = new double[_values.length];
		for (int i = 0; i < _values.length; i++) {
			ret[i] = op.executeScalar(_values[i]);
		}

		return ret;
	}

	protected double[] applyScalarOp(ScalarOperator op, double newVal, int numCols)
		throws DMLRuntimeException 
	{
		//scan over linearized values
		double[] ret = new double[_values.length + numCols];
		for( int i = 0; i < _values.length; i++ ) {
			ret[i] = op.executeScalar(_values[i]);
		}
		
		//add new value to the end
		Arrays.fill(ret, _values.length, _values.length+numCols, newVal);
		
		return ret;
	}
	
	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result) 
		throws DMLRuntimeException 
	{
		unaryAggregateOperations(op, result, 0, getNumRows());
	}
	
	/**
	 * 
	 * @param op aggregation operator
	 * @param result output matrix block
	 * @param rl row lower index, inclusive
	 * @param ru row upper index, exclusive
	 * @throws DMLRuntimeException on invalid inputs
	 */
	public abstract void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result, int rl, int ru)
		throws DMLRuntimeException;
}
