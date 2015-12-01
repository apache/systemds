/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.instructions.spark.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;

import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.FastBufferedDataInputStream;
import org.apache.sysml.runtime.util.FastBufferedDataOutputStream;


public class RowMatrixBlock implements Externalizable
{
	private static final long serialVersionUID = 3307673667643705793L;
	
	private int _len = -1; // block num rows
	private int _row = -1; // in-block row pos
	private MatrixBlock _value = null;
	
	public RowMatrixBlock() {
		//do nothing (required for Externalizable)
	}
	
	public RowMatrixBlock( int len, int row, MatrixBlock value ) {
		_len = len;
		_row = row;
		_value = value;
	}
	
	public int getLen(){
		return _len;
	}
	
	public int getRow(){
		return _row;
	}
	
	public MatrixBlock getValue(){
		return _value;
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
		dos.writeInt(_len);
		dos.writeInt(_row);
		_value.write(dos);
	}
	
	/**
	 * 
	 * @param dis
	 * @throws IOException 
	 */
	private void readHeaderAndPayload(DataInput dis) 
		throws IOException 
	{
		_len = dis.readInt();
		_row = dis.readInt();
		_value = new MatrixBlock();
		_value.readFields(dis);
	}
}
