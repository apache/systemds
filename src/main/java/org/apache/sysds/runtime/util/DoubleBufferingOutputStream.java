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
 
package org.apache.sysds.runtime.util;

import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import org.apache.commons.lang3.concurrent.ConcurrentUtils;

public class DoubleBufferingOutputStream extends FilterOutputStream {
	protected Future<?>[] _locks;
	protected byte[][] _buff;
	private int _pos;
	
	public DoubleBufferingOutputStream(OutputStream out) {
		this(out, 2, 8192);
	}

	public DoubleBufferingOutputStream(OutputStream out, int num, int size) {
		super(out);
		if(size <= 0)
			throw new IllegalArgumentException("Buffer size <= 0.");
		if( size%8 != 0 )
			throw new IllegalArgumentException("Buffer size not a multiple of 8.");
		_buff = new byte[num][size];
		_locks = new Future<?>[num];
		for(int i=0; i<num; i++)
			_locks[i] = ConcurrentUtils.constantFuture(null);
	}

	@Override
	public void write(int b) throws IOException {
		throw new IOException("Not supported"); 
	}

	@Override
	public void write(byte[] b, int off, int len) 
		throws IOException 
	{
		try {
			synchronized(_buff) {
				//block until buffer is free to use
				_locks[_pos].get();
				
				//copy for asynchronous write because b is reused higher up
				System.arraycopy(b, off, _buff[_pos], 0, len);
				
				//submit write request 
				_locks[_pos] = CommonThreadPool.getDynamicPool().submit(new WriteTask(_buff[_pos], len));
				_pos = (_pos+1) % _buff.length;
			}
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
	}
	
	public void writeBuffer(byte[] b, int off, int len) {
		try {
			out.write(b, off, len);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}

	@Override
	public void flush() throws IOException {
		try {
			synchronized(_buff) {
				for(int i=0; i<_buff.length; i++)
					_locks[i].get();
			}
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
		out.flush();
	}

	@Override
	public void close() throws IOException {
		super.close();
	}
	
	private class WriteTask implements Callable<Object> {
		private final byte[] _b;
		private final int _len;
		
		protected WriteTask(byte[] buff, int len) {
			_b = buff;
			_len = len;
		}
		
		@Override
		public Object call() {
			writeBuffer(_b, 0, _len);
			return null;
		}
	}
}
