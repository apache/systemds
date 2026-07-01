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

package org.apache.sysds.runtime.io;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.types.DataType;

/**
 * Single-threaded native Delta Lake reader for frames, built on the Spark-free
 * Delta Kernel library. It opens the latest snapshot of a Delta table, reads
 * its parquet data files through the kernel's default engine (honoring deletion
 * vectors), and materializes the columns into a {@link FrameBlock} whose schema
 * and column names are derived from the Delta table schema.
 *
 * <p>Data is extracted column-at-a-time into primitive arrays (no per-cell
 * boxing or {@code FrameBlock.set} dispatch) and the frame is constructed
 * directly from typed column {@link Array}s. Supported column types map to
 * SystemDS value types: double, float, long, int, short, byte, boolean, and
 * string. Neither the schema nor the dimensions need to be supplied; they are
 * discovered from the table.</p>
 */
public class FrameReaderDelta extends FrameReader {

	//per-column read codes (how to pull a value out of the Delta column vector);
	//aliases of the shared codes in DeltaKernelUtils so the frame read dispatch stays
	//in lockstep with the matrix reader's type mapping. Package visible so the parallel
	//reader can reuse the same dispatch.
	static final int R_DOUBLE = DeltaKernelUtils.T_DOUBLE, R_FLOAT = DeltaKernelUtils.T_FLOAT,
		R_LONG = DeltaKernelUtils.T_LONG, R_INT = DeltaKernelUtils.T_INT, R_SHORT = DeltaKernelUtils.T_SHORT,
		R_BYTE = DeltaKernelUtils.T_BYTE, R_BOOLEAN = DeltaKernelUtils.T_BOOLEAN, R_STRING = DeltaKernelUtils.T_STRING;

	@Override
	public FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException
	{
		Engine engine = DeltaKernelUtils.createEngine();
		String tablePath = DeltaKernelUtils.qualify(fname);

		//per-batch, per-column extracted arrays (boxing free)
		ArrayList<Object[]> batchCols = new ArrayList<>();
		ArrayList<Integer> batchSizes = new ArrayList<>();
		int[] nrowH = new int[1];
		ValueType[][] vtH = new ValueType[1][];
		String[][] nameH = new String[1][];
		int[][] readCodeH = new int[1][];

		DeltaKernelUtils.scan(engine, tablePath, sch -> {
			int ncol = sch.length();
			int[] readCode = new int[ncol];
			ValueType[] vt = new ValueType[ncol];
			String[] cnames = new String[ncol];
			for( int c=0; c<ncol; c++ ) {
				DataType dt = sch.at(c).getDataType();
				readCode[c] = readCode(dt, sch.at(c).getName());
				vt[c] = valueType(readCode[c]);
				cnames[c] = sch.at(c).getName();
			}
			vtH[0] = vt;
			nameH[0] = cnames;
			readCodeH[0] = readCode;
			return (cols, size, selected) -> {
				int n = DeltaKernelUtils.countSelected(size, selected);
				Object[] extracted = new Object[ncol];
				for( int c=0; c<ncol; c++ )
					extracted[c] = extractColumn(cols[c], size, selected, n, readCode[c]);
				batchCols.add(extracted);
				batchSizes.add(n);
				nrowH[0] += n;
			};
		});

		ValueType[] vt = vtH[0];
		String[] discoveredNames = nameH[0];
		int ncol = vt.length;
		int nrow = nrowH[0];

		//empty table: the typed column arrays cannot be zero-length, so return a
		//schema-only frame with the discovered schema/names and zero rows.
		if( nrow == 0 )
			return new FrameBlock(vt, discoveredNames, 0);

		//concatenate the per-batch column arrays into one typed array per column
		Array<?>[] columns = new Array<?>[ncol];
		for( int c=0; c<ncol; c++ )
			columns[c] = buildColumn(vt[c], nrow, batchCols, batchSizes, c);

		FrameBlock ret = new FrameBlock(columns);
		ret.setColumnNames(discoveredNames);
		return ret;
	}

	static Object extractColumn(ColumnVector col, int size, boolean[] selected, int n, int readCode) {
		switch( readCode ) {
			case R_DOUBLE: {
				double[] a = new double[n];
				int lr = 0;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getDouble(r);
					lr++;
				}
				return a;
			}
			case R_FLOAT: {
				float[] a = new float[n];
				int lr = 0;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getFloat(r);
					lr++;
				}
				return a;
			}
			case R_LONG: {
				long[] a = new long[n];
				int lr = 0;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getLong(r);
					lr++;
				}
				return a;
			}
			case R_INT: {
				int[] a = new int[n];
				int lr = 0;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getInt(r);
					lr++;
				}
				return a;
			}
			case R_SHORT: {
				int[] a = new int[n];
				int lr = 0;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getShort(r);
					lr++;
				}
				return a;
			}
			case R_BYTE: {
				int[] a = new int[n];
				int lr = 0;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getByte(r);
					lr++;
				}
				return a;
			}
			case R_BOOLEAN: {
				boolean[] a = new boolean[n];
				int lr = 0;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getBoolean(r);
					lr++;
				}
				return a;
			}
			default: { // R_STRING
				String[] a = new String[n];
				int lr = 0;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					a[lr] = col.isNullAt(r) ? null : col.getString(r);
					lr++;
				}
				return a;
			}
		}
	}

	static Array<?> buildColumn(ValueType vt, int nrow, ArrayList<Object[]> batchCols,
		ArrayList<Integer> batchSizes, int c)
	{
		switch( vt ) {
			case FP64: {
				double[] all = new double[nrow];
				int off = 0;
				for( int b=0; b<batchCols.size(); b++ ) {
					int n = batchSizes.get(b);
					System.arraycopy(batchCols.get(b)[c], 0, all, off, n);
					off += n;
				}
				return ArrayFactory.create(all);
			}
			case FP32: {
				float[] all = new float[nrow];
				int off = 0;
				for( int b=0; b<batchCols.size(); b++ ) {
					int n = batchSizes.get(b);
					System.arraycopy(batchCols.get(b)[c], 0, all, off, n);
					off += n;
				}
				return ArrayFactory.create(all);
			}
			case INT64: {
				long[] all = new long[nrow];
				int off = 0;
				for( int b=0; b<batchCols.size(); b++ ) {
					int n = batchSizes.get(b);
					System.arraycopy(batchCols.get(b)[c], 0, all, off, n);
					off += n;
				}
				return ArrayFactory.create(all);
			}
			case INT32: {
				int[] all = new int[nrow];
				int off = 0;
				for( int b=0; b<batchCols.size(); b++ ) {
					int n = batchSizes.get(b);
					System.arraycopy(batchCols.get(b)[c], 0, all, off, n);
					off += n;
				}
				return ArrayFactory.create(all);
			}
			case BOOLEAN: {
				boolean[] all = new boolean[nrow];
				int off = 0;
				for( int b=0; b<batchCols.size(); b++ ) {
					int n = batchSizes.get(b);
					System.arraycopy(batchCols.get(b)[c], 0, all, off, n);
					off += n;
				}
				return ArrayFactory.create(all);
			}
			default: { // STRING
				String[] all = new String[nrow];
				int off = 0;
				for( int b=0; b<batchCols.size(); b++ ) {
					int n = batchSizes.get(b);
					System.arraycopy(batchCols.get(b)[c], 0, all, off, n);
					off += n;
				}
				return ArrayFactory.create(all);
			}
		}
	}

	static int readCode(DataType dt, String name) {
		//reuse the shared Delta type -> code mapping; frames additionally reject the
		//types the matrix reader also cannot map (typeCode returns -1)
		int code = DeltaKernelUtils.typeCode(dt);
		if( code < 0 )
			throw new DMLRuntimeException("Unsupported non-mappable Delta column '" + name
				+ "' of type " + dt + " for frame read.");
		return code;
	}

	static ValueType valueType(int readCode) {
		switch( readCode ) {
			case R_DOUBLE:  return ValueType.FP64;
			case R_FLOAT:   return ValueType.FP32;
			case R_LONG:    return ValueType.INT64;
			case R_INT:
			case R_SHORT:
			case R_BYTE:    return ValueType.INT32;
			case R_BOOLEAN: return ValueType.BOOLEAN;
			default:        return ValueType.STRING;
		}
	}

	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException
	{
		throw new UnsupportedOperationException(
			"Reading a Delta table from an input stream is not supported; Delta is a directory-based table format.");
	}
}
