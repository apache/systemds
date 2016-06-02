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
import org.apache.sysml.runtime.matrix.data.InputInfo;

/**
 * 
 * 
 */
public class FrameReaderFactory 
{
	/**
	 * 
	 * @param iinfo
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static FrameReader createFrameReader( InputInfo iinfo ) 
		throws DMLRuntimeException
	{
		FileFormatProperties props = (iinfo==InputInfo.CSVInputInfo) ?
			new CSVFileFormatProperties() : null;		
		
		return createFrameReader(iinfo, props);
	}
	
	/**
	 * 
	 * @param props
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static FrameReader createFrameReader( ReadProperties rprops ) 
		throws DMLRuntimeException
	{
		//check valid read properties
		if( rprops == null )
			throw new DMLRuntimeException("Failed to create frame reader with empty properties.");
		
		InputInfo iinfo = rprops.inputInfo;
		FileFormatProperties props = (iinfo==InputInfo.CSVInputInfo) ? ((rprops.formatProperties!=null) ? 
			(CSVFileFormatProperties)rprops.formatProperties : new CSVFileFormatProperties()) : null;		
			
		return createFrameReader(iinfo, props);
	}

	
	/**
	 * 
	 * @param props
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static FrameReader createFrameReader( InputInfo iinfo, FileFormatProperties props ) 
		throws DMLRuntimeException
	{
		FrameReader reader = null;

		if( iinfo == InputInfo.TextCellInputInfo ) {
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_BINARYFORMATS) )
				reader = new FrameReaderTextCellParallel();
			else	
				reader = new FrameReaderTextCell();
		}
		else if( iinfo == InputInfo.CSVInputInfo ) {
			if( props!=null && !(props instanceof CSVFileFormatProperties) )
				throw new DMLRuntimeException("Wrong type of file format properties for CSV writer.");
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_BINARYFORMATS) )
				reader = new FrameReaderTextCSVParallel( (CSVFileFormatProperties)props );
			else
				reader = new FrameReaderTextCSV( (CSVFileFormatProperties)props );
		}
		else if( iinfo == InputInfo.BinaryBlockInputInfo || iinfo == InputInfo.BinaryBlockFrameInputInfo ) {
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_BINARYFORMATS) )
				reader = new FrameReaderBinaryBlockParallel();
			else
				reader = new FrameReaderBinaryBlock();
		}
		else {
			throw new DMLRuntimeException("Failed to create frame reader for unknown input info: "
		                                   + InputInfo.inputInfoToString(iinfo));
		}
		
		return reader;
	}
}
