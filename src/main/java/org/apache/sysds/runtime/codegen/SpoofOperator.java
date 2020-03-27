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

package org.apache.sysds.runtime.codegen;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;

public abstract class SpoofOperator implements Serializable
{
	private static final long serialVersionUID = 3834006998853573319L;
	private static final Log LOG = LogFactory.getLog(SpoofOperator.class.getName());
	
	protected static final long PAR_NUMCELL_THRESHOLD = 1024*1024;   //Min 1M elements
	protected static final long PAR_MINFLOP_THRESHOLD = 2L*1024*1024; //MIN 2 MFLOP
	
	
	public abstract MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars, MatrixBlock out);
	
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars, MatrixBlock out, int k) {
		//default implementation serial execution
		return execute(inputs, scalars, out);
	}
	
	public abstract String getSpoofType();
	
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars) {
		throw new DMLRuntimeException("Invalid invocation in base class.");
	}
	
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars, int k) {
		//default implementation serial execution
		return execute(inputs, scalars);
	}
	
	protected SideInput[] prepInputMatrices(ArrayList<MatrixBlock> inputs) {
		return prepInputMatrices(inputs, 1, inputs.size()-1, false, false);
	}
	
	protected SideInput[] prepInputMatrices(ArrayList<MatrixBlock> inputs, boolean denseOnly) {
		return prepInputMatrices(inputs, 1, inputs.size()-1, denseOnly, false);
	}
	
	protected SideInput[] prepInputMatrices(ArrayList<MatrixBlock> inputs, int offset, boolean denseOnly) {
		return prepInputMatrices(inputs, offset, inputs.size()-offset, denseOnly, false);
	}
	
	protected SideInput[] prepInputMatrices(ArrayList<MatrixBlock> inputs, boolean denseOnly, boolean tB1) {
		return prepInputMatrices(inputs, 1, inputs.size()-1, denseOnly, tB1);
	}
	
	protected SideInput[] prepInputMatrices(ArrayList<MatrixBlock> inputs, int offset, int len, boolean denseOnly, boolean tB1)
	{
		SideInput[] b = new SideInput[len];
		for(int i=offset; i<offset+len; i++) {
			//transpose if necessary
			int clen = inputs.get(i).getNumColumns();
			MatrixBlock in = (tB1 && i==1 ) ? LibMatrixReorg.transpose(inputs.get(i), 
				new MatrixBlock(clen, inputs.get(i).getNumRows(), false)) : inputs.get(i);
			
			//create side input
			if( denseOnly && (in.isInSparseFormat() || !in.isAllocated()) ) {
				//convert empty or sparse to dense temporary block (note: we don't do
				//this in place because this block might be used by multiple threads)
				if( in.getNumColumns()==1 && in.isEmptyBlock(false) ) //dense empty
					b[i-offset] = new SideInput(null, null, clen);
				else {
					b[i-offset] = new SideInput(DataConverter.convertToDenseBlock(in, false), null, clen);
					LOG.warn(getClass().getName()+": Converted "+in.getNumRows()+"x"+in.getNumColumns()+
						", nnz="+in.getNonZeros()+" sideways input matrix from sparse to dense.");
				}
			}
			else if( in.isInSparseFormat() || !in.isAllocated() ) {
				b[i-offset] = new SideInput(null, in, clen);
			}
			else {
				b[i-offset] = new SideInput(in.getDenseBlock(), null, clen);
			}
		}
		
		return b;
	}
	
	protected static SideInput[] createSparseSideInputs(SideInput[] input) {
		return createSparseSideInputs(input, false);
	}
	
	protected static SideInput[] createSparseSideInputs(SideInput[] input, boolean row) {
		//determine if there are sparse side inputs
		boolean containsSparse = false;
		for( int i=0; i<input.length; i++ ) {
			SideInput tmp = input[i];
			containsSparse |= (tmp.mdat != null && tmp.clen > 1);
		}
		if( !containsSparse )
			return input;
		SideInput[] ret = new SideInput[input.length];
		for( int i=0; i<input.length; i++ ) {
			SideInput tmp = input[i];
			ret[i] = (tmp.mdat != null && tmp.clen > 1) ?
				(row ? new SideInputSparseRow(tmp) : 
				new SideInputSparseCell(tmp)) : tmp;
		}
		return ret;
	}
	
	public static DenseBlock[] getDenseMatrices(SideInput[] inputs) {
		DenseBlock[] ret = new DenseBlock[inputs.length];
		for( int i=0; i<inputs.length; i++ )
			ret[i] = inputs[i].ddat;
		return ret;
	}
	
	protected static double[] prepInputScalars(ArrayList<ScalarObject> scalarObjects) {
		double[] scalars = new double[scalarObjects.size()];
		for(int i=0; i < scalarObjects.size(); i++)
			scalars[i] = scalarObjects.get(i).getDoubleValue();
		return scalars;
	}
	
	public static long getTotalInputNnz(ArrayList<MatrixBlock> inputs) {
		return inputs.stream().mapToLong(in -> in.getNonZeros()).sum();
	}
	
	public static long getTotalInputSize(ArrayList<MatrixBlock> inputs) {
		return inputs.stream().mapToLong(
			in -> (long)in.getNumRows() * in.getNumColumns()).sum();
	}
	
	//abstraction for safely accessing sideways matrices without the need 
	//to allocate empty matrices as dense, see prepInputMatrices
	
	protected static double getValue(double[] data, double index) {
		int iindex = UtilFunctions.toInt(index);
		return getValue(data, iindex);
	}
	
	protected static double getValue(double[] data, int index) {
		return (data!=null) ? data[index] : 0;
	}
	
	protected static double getValue(double[] data, int n, double rowIndex, double colIndex) {
		int irowIndex = UtilFunctions.toInt(rowIndex);
		int icolIndex = UtilFunctions.toInt(colIndex);
		return getValue(data, n, irowIndex, icolIndex);
	}
	
	protected static double getValue(double[] data, int n, int rowIndex, int colIndex) {
		return (data!=null) ? data[rowIndex*n+colIndex] : 0;
	}
	
	protected static double getValue(double[] avals, int[] aix, int ai, int alen, double colIndex) {
		int icolIndex = UtilFunctions.toInt(colIndex);
		return getValue(avals, aix, ai, alen, icolIndex);
	}
	
	protected static double getValue(double[] avals, int[] aix, int ai, int alen, int colIndex) {
		int pos = Arrays.binarySearch(aix, ai, ai+alen, colIndex);
		return (pos >= 0) ? avals[pos] : 0;
	}
	
	protected static double getValue(SideInput data, double rowIndex) {
		int irowIndex = UtilFunctions.toInt(rowIndex);
		return getValue(data, irowIndex);
	}
	
	protected static double getValue(SideInput data, int rowIndex) {
		//note: wrapper sideinput guaranteed to exist
		return (data.ddat!=null) ? data.ddat.valuesAt(0)[rowIndex] :
			(data.mdat!=null) ? data.mdat.quickGetValue(rowIndex, 0) : 0;
	}
	
	protected static double getValue(SideInput data, int n, double rowIndex, double colIndex) {
		int irowIndex = UtilFunctions.toInt(rowIndex);
		int icolIndex = UtilFunctions.toInt(colIndex);
		return getValue(data, n, irowIndex, icolIndex);
	}
	
	protected static double getValue(SideInput data, int n, int rowIndex, int colIndex) {
		//note: wrapper sideinput guaranteed to exist
		return (data.ddat!=null) ? data.ddat.get(rowIndex, colIndex) :
			(data instanceof SideInputSparseCell) ? 
			((SideInputSparseCell)data).next(rowIndex, colIndex) :
			(data.mdat!=null) ? data.mdat.quickGetValue(rowIndex, colIndex) : 0;
	}
	
	protected static double[] getVector(SideInput data, int n, double rowIndex, double colIndex) {
		int irowIndex = UtilFunctions.toInt(rowIndex);
		int icolIndex = UtilFunctions.toInt(colIndex);
		return getVector(data, n, irowIndex, icolIndex);
	}
	
	protected static double[] getVector(SideInput data, int n, int rowIndex, int colIndex) {
		double[] c = LibSpoofPrimitives.allocVector(colIndex+1, false);
		System.arraycopy(data.values(rowIndex), data.pos(rowIndex), c, 0, colIndex+1);
		return c;
	}
	
	public static class SideInput {
		public final DenseBlock ddat;
		public final MatrixBlock mdat;
		public final int clen;
		public SideInput(DenseBlock ddata, MatrixBlock mdata, int clength) {
			ddat = ddata;
			mdat = mdata;
			clen = clength;
		}
		public int pos(int r) {
			return (ddat!=null) ? ddat.pos(r) : r * clen;
		}
		public double[] values(int r) {
			return (ddat!=null) ? ddat.values(r) : null;
		}
		public double getValue(int r, int c) {
			return SpoofOperator.getValue(this, clen, r, c);
		}
		public void reset() {}
	}
	
	public static class SideInputSparseRow extends SideInput {
		private final double[] values;
		private int currRowIndex = -1;
		
		public SideInputSparseRow(SideInput in) {
			super(in.ddat, in.mdat, in.clen);
			values = new double[in.clen];
		}
		@Override
		public int pos(int r) {
			return 0;
		}
		@Override
		public double[] values(int r) {
			if( r > currRowIndex )
				nextRow(r);
			return values;
		}
		
		private void nextRow(int r) {
			currRowIndex = r;
			SparseBlock sblock = mdat.getSparseBlock();
			if( sblock == null ) return;
			Arrays.fill(values, 0);
			if( !sblock.isEmpty(r) ) {
				int apos = sblock.pos(r);
				int alen = sblock.size(r);
				int[] aix = sblock.indexes(r);
				double[] avals = sblock.values(r);
				for(int k=apos; k<apos+alen; k++)
					values[aix[k]] = avals[k];
			}
		}
	}
	
	public static class SideInputSparseCell extends SideInput {
		private int currRowIndex = -1;
		private int currColPos = 0;
		private int currLen = 0;
		private int[] indexes;
		private double[] values;
		
		public SideInputSparseCell(SideInput in) {
			super(in.ddat, in.mdat, in.clen);
		}
		public double next(int rowIndex, int colIndex) {
			SparseBlock sblock = mdat.getSparseBlock();
			if( sblock == null || sblock.isEmpty(rowIndex) )
				return 0;
			//move to next row if necessary
			if( rowIndex > currRowIndex ) {
				currRowIndex = rowIndex;
				currColPos = sblock.pos(currRowIndex);
				currLen = sblock.size(currRowIndex) + currColPos;
				indexes = sblock.indexes(currRowIndex);
				values = sblock.values(currRowIndex);
			}
			//move to next colpos if necessary
			while( currColPos < currLen && indexes[currColPos]<colIndex )
				currColPos ++;
			//return current value or zero
			return (currColPos < currLen && indexes[currColPos]==colIndex) ?
				values[currColPos] : 0;
		}
		@Override
		public void reset() {
			currColPos = 0;
		}
	}
}
