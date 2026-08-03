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

package org.apache.sysds.runtime.ooc.cache.io;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlockDataInput;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

class OOCBufferedDataInputStream implements DataInput, MatrixBlockDataInput {
	private static final int PAGE_SIZE = 4096;
	private static final int PAGE_MASK = PAGE_SIZE - 1;
	private static final int DEFAULT_BUFFER_SIZE = 64 * 1024;
	private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
	private static final int DOUBLE_VECTOR_LENGTH = DOUBLE_SPECIES.length();

	private final RandomAccessFile _in;
	private final byte[] _buff;
	private final byte[] _tmp;
	private final DoubleBuffer[] _doubleDecodeBuffers;
	private final int _bufflen;
	private long _filePos;
	private int _pos;
	private int _count;

	OOCBufferedDataInputStream(RandomAccessFile in) throws IOException {
		this(in, DEFAULT_BUFFER_SIZE);
	}

	OOCBufferedDataInputStream(RandomAccessFile in, int size) throws IOException {
		if(size <= 0)
			throw new IllegalArgumentException("Buffer size <= 0.");
		if(size % 8 != 0)
			throw new IllegalArgumentException("Buffer size not a multiple of 8.");
		_in = in;
		_buff = new byte[size];
		_tmp = new byte[8];
		_doubleDecodeBuffers = createDoubleDecodeBuffers(_buff);
		_bufflen = size;
		_filePos = in.getFilePointer();
		_pos = 0;
		_count = 0;
	}

	@Override
	public void readFully(byte[] b) throws IOException {
		readFully(b, 0, b.length);
	}

	@Override
	public void readFully(byte [] b, int off, int len) throws IOException {
		if(len < 0)
			throw new IndexOutOfBoundsException();

		while(len > 0) {
			int avail = _count - _pos;
			if(avail > 0) {
				int n = Math.min(avail, len);
				System.arraycopy(_buff, _pos, b, off, n);
				_pos += n;
				off += n;
				len -= n;
			}
			else if(len >= _bufflen) {
				_in.readFully(b, off, len);
				_filePos += len;
				return;
			}
			else {
				refill();
			}
		}
	}

	@Override
	public int skipBytes(int n) throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public boolean readBoolean() throws IOException {
		return readByte() != 0;
	}

	@Override
	public byte readByte() throws IOException {
		if(_pos >= _count)
			refill();
		return _buff[_pos++];
	}

	@Override
	public int readUnsignedByte() throws IOException {
		return readByte() & 0xFF;
	}

	@Override
	public short readShort() throws IOException {
		if(_count - _pos >= 2) {
			short ret = (short)baToShort(_buff, _pos);
			_pos += 2;
			return ret;
		}
		readFully(_tmp, 0, 2);
		return (short)baToShort(_tmp, 0);
	}

	@Override
	public int readUnsignedShort() throws IOException {
		return readShort() & 0xFFFF;
	}

	@Override
	public char readChar() throws IOException {
		return (char)readUnsignedShort();
	}

	@Override
	public int readInt() throws IOException {
		if(_count - _pos >= 4) {
			int ret = baToInt(_buff, _pos);
			_pos += 4;
			return ret;
		}
		readFully(_tmp, 0, 4);
		return baToInt(_tmp, 0);
	}

	@Override
	public long readLong() throws IOException {
		if(_count - _pos >= 8) {
			long ret = baToLong(_buff, _pos);
			_pos += 8;
			return ret;
		}
		readFully(_tmp, 0, 8);
		return baToLong(_tmp, 0);
	}

	@Override
	public float readFloat() throws IOException {
		return Float.intBitsToFloat(readInt());
	}

	@Override
	public double readDouble() throws IOException {
		return Double.longBitsToDouble(readLong());
	}

	@Override
	public String readLine() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public String readUTF() throws IOException {
		return DataInputStream.readUTF(this);
	}

	@Override
	public long readDoubleArray(int len, double[] varr) throws IOException {
		if(len <= 0 || len > varr.length)
			throw new IndexOutOfBoundsException("len=" + len + ", varr.length=" + varr.length);

		long nnz = 0;
		int ix = 0;
		while(ix < len) {
			int avail = _count - _pos;
			if(avail <= 0) {
				refill();
				continue;
			}
			if(avail < 8) {
				readFully(_tmp, 0, 8);
				double v = Double.longBitsToDouble(baToLong(_tmp, 0));
				varr[ix] = v;
				nnz += (v != 0) ? 1 : 0;
				ix++;
				continue;
			}

			int ndbl = Math.min(len - ix, avail / 8);
			int end = _pos + ndbl * 8;
			readDoubles(_pos, ndbl, varr, ix);
			nnz += countNonZeros(varr, ix, ndbl);
			ix += ndbl;
			_pos = end;
		}
		return nnz;
	}

	@Override
	public long readSparseRows(int rlen, long nnz, SparseBlock rows) throws IOException {
		if(rows instanceof SparseBlockCSR) {
			((SparseBlockCSR)rows).initSparse(rlen, (int)nnz, this);
			return nnz;
		}

		long gnnz = 0;
		for(int i = 0; i < rlen; i++) {
			int lnnz = readInt();
			if(lnnz > 0) {
				rows.allocate(i, lnnz);

				for(int j = 0; j < lnnz; j++) {
					int aix = readInt();
					double aval = readDouble();
					rows.append(i, aix, aval);
				}
				gnnz += lnnz;
			}
		}

		if(gnnz != nnz)
			throw new IOException("Invalid number of read nnz: " + gnnz + " vs " + nnz);
		return nnz;
	}

	private void refill() throws IOException {
		int len = getRefillLength();
		_count = _in.read(_buff, 0, len);
		_pos = 0;
		if(_count < 0)
			throw new EOFException();
		_filePos += _count;
	}

	private int getRefillLength() {
		int pageOffset = (int)(_filePos & PAGE_MASK);
		if(pageOffset == 0)
			return _bufflen;
		return Math.min(_bufflen, PAGE_SIZE - pageOffset);
	}

	private static int baToShort(byte[] ba, final int off) {
		return IOUtilFunctions.baToShort(ba, off);
	}

	private static int baToInt(byte[] ba, final int off) {
		return IOUtilFunctions.baToInt(ba, off);
	}

	private static long baToLong(byte[] ba, final int off) {
		return IOUtilFunctions.baToLong(ba, off);
	}

	private void readDoubles(int srcPos, int len, double[] dest, int destPos) {
		int alignment = srcPos & 7;
		DoubleBuffer dbuff = _doubleDecodeBuffers[alignment];
		dbuff.position((srcPos - alignment) >>> 3);
		dbuff.get(dest, destPos, len);
	}

	private static DoubleBuffer[] createDoubleDecodeBuffers(byte[] buff) {
		DoubleBuffer[] ret = new DoubleBuffer[8];
		for(int i = 0; i < ret.length; i++) {
			ByteBuffer bbuff = ByteBuffer.wrap(buff);
			bbuff.position(i);
			ret[i] = bbuff.slice().order(ByteOrder.BIG_ENDIAN).asDoubleBuffer();
		}
		return ret;
	}

	private static long countNonZeros(double[] values, int off, int len) {
		long nnz = 0;
		int i = 0;
		int upper = DOUBLE_SPECIES.loopBound(len);
		DoubleVector vzero = DoubleVector.zero(DOUBLE_SPECIES);
		for(; i < upper; i += DOUBLE_VECTOR_LENGTH) {
			DoubleVector v = DoubleVector.fromArray(DOUBLE_SPECIES, values, off + i);
			nnz += v.compare(VectorOperators.NE, vzero).trueCount();
		}
		for(; i < len; i++)
			nnz += (values[off + i] != 0) ? 1 : 0;
		return nnz;
	}
}
