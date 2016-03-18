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

package org.apache.sysml.runtime.io;

import java.io.IOException;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

/**
 * 
 */
public class FrameWriterTextCSV extends FrameWriter
{
	//blocksize for string concatenation in order to prevent write OOM 
	//(can be set to very large value to disable blocking)
	public static final int BLOCKSIZE_J = 32; //32 cells (typically ~512B, should be less than write buffer of 1KB)
	
	protected CSVFileFormatProperties _props = null;
	
	public FrameWriterTextCSV( CSVFileFormatProperties props ) {
		_props = props;
	}
	
	/**
	 * @param src
	 * @param fname
	 * @return
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	@Override
	public void writeFrameToHDFS( FrameBlock src, String fname )
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
	}

}
