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

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.OutputInfo;

/**
 * 
 * 
 */
public class FrameWriterFactory 
{
	/**
	 * 
	 * @param oinfo
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static FrameWriter createFrameWriter( OutputInfo oinfo ) 
			throws DMLRuntimeException
	{
		return createFrameWriter(oinfo, null);
	}
	
	/**
	 * 
	 * @param oinfo
	 * @param props 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static FrameWriter createFrameWriter( OutputInfo oinfo, FileFormatProperties props ) 
		throws DMLRuntimeException
	{
		FrameWriter writer = null;
		
		if( oinfo == OutputInfo.TextCellOutputInfo ) {
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS) )
				writer = new FrameWriterTextCellParallel();
			else
				writer = new FrameWriterTextCell();
		}
		else if( oinfo == OutputInfo.CSVOutputInfo ) {
			if( props!=null && !(props instanceof CSVFileFormatProperties) )
				throw new DMLRuntimeException("Wrong type of file format properties for CSV writer.");
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS) )
				writer = new FrameWriterTextCSVParallel((CSVFileFormatProperties)props);
			else
				writer = new FrameWriterTextCSV((CSVFileFormatProperties)props);	
		}
		else if( oinfo == OutputInfo.BinaryBlockOutputInfo ) {
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS) )
				writer = new FrameWriterBinaryBlockParallel();
			else
				writer = new FrameWriterBinaryBlock();
		}
		else {
			throw new DMLRuntimeException("Failed to create frame writer for unknown output info: "
		                                   + OutputInfo.outputInfoToString(oinfo));
		}
		
		return writer;
	}
	
}
