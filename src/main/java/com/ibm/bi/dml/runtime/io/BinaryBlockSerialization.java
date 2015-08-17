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

package com.ibm.bi.dml.runtime.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.apache.hadoop.io.serializer.Deserializer;
import org.apache.hadoop.io.serializer.Serialization;
import org.apache.hadoop.io.serializer.Serializer;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.FastBufferedDataInputStream;
import com.ibm.bi.dml.runtime.util.FastBufferedDataOutputStream;

/**
 * This custom serialization class can be used via 
 * job.set("io.serializations", "com.ibm.bi.dml.runtime.io.BinaryBlockSerialization"); 
 * 
 * 
 */
@SuppressWarnings("rawtypes")
public class BinaryBlockSerialization implements Serialization
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public boolean accept(Class arg0) 
	{
		boolean ret = false;
		
		if( //arg0 == MatrixIndexes.class || //binary block key
			arg0 == MatrixBlock.class  ) //binary block 
		{
			ret = true;
		}
		
		return ret;
	}

	@Override
	public Deserializer<MatrixBlock> getDeserializer(Class arg0) 
	{
		return new MatrixBlockDeserializer();
	}

	@Override
	public Serializer<MatrixBlock> getSerializer(Class arg0) 
	{
		return new MatrixBlockSerializer();
	}

	/**
	 * 
	 * 
	 */
	public static class MatrixBlockDeserializer implements Deserializer<MatrixBlock>
	{
		private FastBufferedDataInputStream _in = null; 
		private MatrixBlock _buff = null;
		
		@Override
		public void open(InputStream arg0) 
			throws IOException 
		{
			_in = new FastBufferedDataInputStream( arg0 );	
		}
	
		@Override
		public MatrixBlock deserialize(MatrixBlock mb) 
			throws IOException 
		{
			//internal buffer usage for robustness (if required)
			if( mb == null ){
				if( _buff == null )
					_buff = new MatrixBlock();
				mb = _buff;
			}
			
			//core deserialize
			mb.readFields(_in);
			
			return mb;
		}

		@Override
		public void close() 
			throws IOException 
		{
			if( _in != null )
				_in.close();
		}
	}
	
	/**
	 * 
	 * 
	 */
	public static class MatrixBlockSerializer implements Serializer<MatrixBlock>
	{
		private FastBufferedDataOutputStream _out = null;
		
		@Override
		public void open(OutputStream arg0) 
			throws IOException 
		{
			_out = new FastBufferedDataOutputStream( arg0 );
		}

		@Override
		public void serialize(MatrixBlock mb) 
			throws IOException 
		{
			mb.write( _out );
			
			//flush for guaranteed write (currently required)
			_out.flush();
		}
	
		@Override
		public void close() 
			throws IOException 
		{			
			if( _out != null )
				_out.close();
		}
	}
}
