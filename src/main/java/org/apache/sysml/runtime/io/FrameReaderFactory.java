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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
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
		FrameReader reader = null;
		
		if( iinfo == InputInfo.TextCellInputInfo )
		{
			reader = new FrameReaderTextCell( iinfo );	
		}
		else if( iinfo == InputInfo.CSVInputInfo )
		{
			reader = new FrameReaderTextCSV(new CSVFileFormatProperties());
		}
		else if( iinfo == InputInfo.BinaryBlockInputInfo ) {
			reader = new FrameReaderBinaryBlock( false );
		}
		else {
			throw new DMLRuntimeException("Failed to create frame reader for unknown input info: "
		                                   + InputInfo.inputInfoToString(iinfo));
		}
		
		return reader;
	}
	
	/**
	 * 
	 * @param props
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static FrameReader createFrameReader( ReadProperties props ) 
		throws DMLRuntimeException
	{
		//check valid read properties
		if( props == null )
			throw new DMLRuntimeException("Failed to create frame reader with empty properties.");
		
		FrameReader reader = null;
		InputInfo iinfo = props.inputInfo;

		if( iinfo == InputInfo.TextCellInputInfo ) {
			reader = new FrameReaderTextCell( iinfo );
		}
		else if( iinfo == InputInfo.CSVInputInfo ) {
			reader = new FrameReaderTextCSV( props.formatProperties!=null ? (CSVFileFormatProperties)props.formatProperties : new CSVFileFormatProperties());
		}
		else if( iinfo == InputInfo.BinaryBlockInputInfo ) {
			reader = new FrameReaderBinaryBlock( props.localFS );
		}
		else {
			throw new DMLRuntimeException("Failed to create frame reader for unknown input info: "
		                                   + InputInfo.inputInfoToString(iinfo));
		}
		
		return reader;
	}
}
