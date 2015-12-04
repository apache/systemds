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

package org.apache.sysml.runtime.transform;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;

import org.apache.sysml.runtime.matrix.CSVReblockMR.OffsetCount;
import org.apache.sysml.runtime.util.UtilFunctions;

public class DistinctValue implements Writable, Serializable {
	
	private static final long serialVersionUID = -8236705946336974836L;

	private static final byte [] EMPTY_BYTES = new byte[0];
	  
	// word (distinct value)
	private byte[] _bytes;
	private int _length;
	// count
	private long _count;
	
	public DistinctValue() {
		_bytes = EMPTY_BYTES;
		_length = 0;
		_count = -1;
	}
	
	public DistinctValue(String w, long count) throws CharacterCodingException {
	    ByteBuffer bb = Text.encode(w, true);
	    _bytes = bb.array();
	    _length = bb.limit();
		_count = count;
	}
	
	public DistinctValue(OffsetCount oc) throws CharacterCodingException 
	{
		this(oc.filename + "," + oc.fileOffset, oc.count);
	}
	
	public void reset() {
		_bytes = EMPTY_BYTES;
		_length = 0;
		_count = -1;
	}
	
	public String getWord() {  return new String( _bytes, 0, _length, Charset.forName("UTF-8") ); }
	public long getCount() { return _count; }
	
	@Override
	public void write(DataOutput out) throws IOException {
	    // write word
		WritableUtils.writeVInt(out, _length);
	    out.write(_bytes, 0, _length);
		// write count
	    out.writeLong(_count);
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
	    // read word 
		int newLength = WritableUtils.readVInt(in);
	    _bytes = new byte[newLength];
	    in.readFully(_bytes, 0, newLength);
	    _length = newLength;
	    if (_length != _bytes.length)
	    	System.out.println("ERROR in DistinctValue.readFields()");
	    // read count
	    _count = in.readLong();
	}
	
	public OffsetCount getOffsetCount() {
		OffsetCount oc = new OffsetCount();
		String[] parts = getWord().split(",");
		oc.filename = parts[0];
		oc.fileOffset = UtilFunctions.parseToLong(parts[1]);
		oc.count = getCount();
		
		return oc;
	}
	
}
