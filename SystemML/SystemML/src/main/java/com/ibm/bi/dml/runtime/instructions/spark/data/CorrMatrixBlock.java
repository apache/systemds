/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.FastBufferedDataInputStream;
import com.ibm.bi.dml.runtime.util.FastBufferedDataOutputStream;


public class CorrMatrixBlock implements Externalizable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -2204456681697015083L;
	
	private MatrixBlock _value = null;
	private MatrixBlock _corr = null;
	
	public CorrMatrixBlock() {
		//do nothing (required for Externalizable)
	}
	
	public CorrMatrixBlock( MatrixBlock value ) {
		//shallow copy of passed value
		_value = value;
	}
	
	public CorrMatrixBlock( MatrixBlock value, MatrixBlock corr ) {
		//shallow copy of passed value and corr
		_value = value;
		_corr = corr;
	}
	
	public MatrixBlock getValue(){
		return _value;
	}
	
	public MatrixBlock getCorrection(){
		return _corr;
	}
	
	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient deserialization. 
	 * 
	 * @param is
	 * @throws IOException
	 */
	public void readExternal(ObjectInput is) 
		throws IOException
	{
		DataInput dis = is;
		
		if( is instanceof ObjectInputStream ) {
			//fast deserialize of dense/sparse blocks
			ObjectInputStream ois = (ObjectInputStream)is;
			dis = new FastBufferedDataInputStream(ois);
		}
		
		readHeaderAndPayload(dis);
	}

	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient serialization. 
	 * 
	 * @param is
	 * @throws IOException
	 */
	public void writeExternal(ObjectOutput os) 
		throws IOException
	{
		if( os instanceof ObjectOutputStream ) {
			//fast serialize of dense/sparse blocks
			ObjectOutputStream oos = (ObjectOutputStream)os;
			FastBufferedDataOutputStream fos = new FastBufferedDataOutputStream(oos);
			writeHeaderAndPayload(fos);
			fos.flush();
		}
		else {
			//default serialize (general case)
			writeHeaderAndPayload(os);	
		}
	}

	/**
	 * 
	 * @param dos
	 * @throws IOException
	 */
	private void writeHeaderAndPayload(DataOutput dos) 
		throws IOException 
	{
		dos.writeByte((_corr!=null)?1:0);
		_value.write(dos);
		if( _corr!=null )
			_corr.write(dos);
	}
	
	/**
	 * 
	 * @param dis
	 * @throws IOException 
	 */
	private void readHeaderAndPayload(DataInput dis) 
		throws IOException 
	{
		boolean corrExists = (dis.readByte() != 0) ? true : false;
		_value = new MatrixBlock();
		_value.readFields(dis);
		if( corrExists ) {
			_corr = new MatrixBlock();
			_corr.readFields(dis);	
		}
	}
}
