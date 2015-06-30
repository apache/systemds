package com.ibm.bi.dml.runtime.transform;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;

import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class DistinctValue implements Writable {
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
