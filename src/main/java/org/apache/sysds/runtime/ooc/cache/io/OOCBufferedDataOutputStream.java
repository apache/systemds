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
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlockDataOutput;

import java.io.DataOutput;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UTFDataFormatException;

class OOCBufferedDataOutputStream extends FilterOutputStream implements DataOutput, MatrixBlockDataOutput {
	private final byte[] _buff;
	private final int _bufflen;
	private int _count;
	private long _position;
	private long _flushedPosition;

	OOCBufferedDataOutputStream(OutputStream out) {
		this(out, 8192);
	}

	OOCBufferedDataOutputStream(OutputStream out, int size) {
		super(out);
		if(size <= 0)
			throw new IllegalArgumentException("Buffer size <= 0.");
		if(size % 8 != 0)
			throw new IllegalArgumentException("Buffer size not a multiple of 8.");
		_buff = new byte[size];
		_bufflen = size;
		_count = 0;
		_position = 0;
		_flushedPosition = 0;
	}

	long getPosition() {
		return _position;
	}

	long getFlushedPosition() {
		return _flushedPosition;
	}

	@Override
	public void write(int b) throws IOException {
		if(_count >= _bufflen)
			flushBuffer();
		_buff[_count++] = (byte)b;
		_position++;
	}

	@Override
	public void write(byte[] b, int off, int len) throws IOException {
		if(len > _bufflen) {
			flushBuffer();
			out.write(b, off, len);
			_position += len;
			_flushedPosition += len;
		}
		else {
			if(len > _bufflen - _count)
				flushBuffer();
			System.arraycopy(b, off, _buff, _count, len);
			_count += len;
			_position += len;
		}
	}

	@Override
	public void flush() throws IOException {
		flushBuffer();
		out.flush();
	}

	private void flushBuffer() throws IOException {
		if(_count > 0) {
			out.write(_buff, 0, _count);
			_flushedPosition += _count;
			_count = 0;
		}
	}

	@Override
	public void close() throws IOException {
		super.close();
	}

	@Override
	public void writeBoolean(boolean v) throws IOException {
		if(_count >= _bufflen)
			flushBuffer();
		_buff[_count++] = (byte)(v ? 1 : 0);
		_position++;
	}

	@Override
	public void writeInt(int v) throws IOException {
		if(_count + 4 > _bufflen)
			flushBuffer();
		intToBa(v, _buff, _count);
		_count += 4;
		_position += 4;
	}

	@Override
	public void writeLong(long v) throws IOException {
		if(_count + 8 > _bufflen)
			flushBuffer();
		longToBa(v, _buff, _count);
		_count += 8;
		_position += 8;
	}

	@Override
	public void writeDouble(double v) throws IOException {
		if(_count + 8 > _bufflen)
			flushBuffer();
		longToBa(Double.doubleToRawLongBits(v), _buff, _count);
		_count += 8;
		_position += 8;
	}

	@Override
	public void writeFloat(float v) throws IOException {
		if(_count + 4 > _bufflen)
			flushBuffer();
		intToBa(Float.floatToIntBits(v), _buff, _count);
		_count += 4;
		_position += 4;
	}

	@Override
	public void writeByte(int v) throws IOException {
		if(_count + 1 > _bufflen)
			flushBuffer();
		_buff[_count++] = (byte)v;
		_position++;
	}

	@Override
	public void writeShort(int v) throws IOException {
		if(_count + 2 > _bufflen)
			flushBuffer();
		shortToBa(v, _buff, _count);
		_count += 2;
		_position += 2;
	}

	@Override
	public void writeBytes(String s) throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public void writeChar(int v) throws IOException {
		writeShort(v);
	}

	@Override
	public void writeChars(String s) throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public void writeUTF(String s) throws IOException {
		int slen = s.length();
		int utflen = IOUtilFunctions.getUTFSize(s) - 2;
		if(utflen - 2 > 65535)
			throw new UTFDataFormatException("encoded string too long: " + utflen);

		writeShort(utflen);
		for(int i = 0; i < slen; i++) {
			if(_count + 3 > _bufflen)
				flushBuffer();
			final char c = s.charAt(i);
			if(c >= 0x0001 && c <= 0x007F) {
				_buff[_count++] = (byte)c;
				_position++;
			}
			else if(c >= 0x0800) {
				_buff[_count++] = (byte)(0xE0 | ((c >> 12) & 0x0F));
				_buff[_count++] = (byte)(0x80 | ((c >> 6) & 0x3F));
				_buff[_count++] = (byte)(0x80 | (c & 0x3F));
				_position += 3;
			}
			else {
				_buff[_count++] = (byte)(0xC0 | ((c >> 6) & 0x1F));
				_buff[_count++] = (byte)(0x80 | (c & 0x3F));
				_position += 2;
			}
		}
	}

	@Override
	public void writeDoubleArray(int len, double[] varr) throws IOException {
		for(int i = 0; i < len; ) {
			if(_count >= _bufflen)
				flushBuffer();
			int lblen = Math.min(len - i, (_bufflen - _count) / 8);
			if(lblen == 0) {
				flushBuffer();
				continue;
			}
			for(int j = 0; j < lblen; j++) {
				longToBa(Double.doubleToRawLongBits(varr[i + j]), _buff, _count);
				_count += 8;
			}
			_position += 8L * lblen;
			i += lblen;
			if(_count >= _bufflen)
				flushBuffer();
		}
	}

	@Override
	public void writeSparseRows(int rlen, SparseBlock rows) throws IOException {
		int lrlen = Math.min(rows.numRows(), rlen);
		for(int i = 0; i < lrlen; i++) {
			if(!rows.isEmpty(i)) {
				int apos = rows.pos(i);
				int alen = rows.size(i);
				int[] aix = rows.indexes(i);
				double[] avals = rows.values(i);

				writeInt(alen);

				for(int j = apos; j < apos + alen; j++) {
					if(_count + 12 > _bufflen)
						flushBuffer();
					long tmp = Double.doubleToRawLongBits(avals[j]);
					intToBa(aix[j], _buff, _count);
					longToBa(tmp, _buff, _count + 4);
					_count += 12;
					_position += 12;
				}
			}
			else {
				writeInt(0);
			}
		}

		for(int i = lrlen; i < rlen; i++)
			writeInt(0);
	}

	private static void shortToBa(final int val, byte[] ba, final int off) {
		IOUtilFunctions.shortToBa(val, ba, off);
	}

	private static void intToBa(final int val, byte[] ba, final int off) {
		IOUtilFunctions.intToBa(val, ba, off);
	}

	private static void longToBa(final long val, byte[] ba, final int off) {
		IOUtilFunctions.longToBa(val, ba, off);
	}
}
