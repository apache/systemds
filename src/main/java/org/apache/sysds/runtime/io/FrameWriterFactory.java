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

package org.apache.sysds.runtime.io;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.runtime.DMLRuntimeException;

public class FrameWriterFactory 
{
	public static FrameWriter createFrameWriter(FileFormat fmt) {
		return createFrameWriter(fmt, null);
	}

	public static FrameWriter createFrameWriter( FileFormat fmt, FileFormatProperties props ) {
		FrameWriter writer = null;
		switch(fmt) {
			case TEXT:
				if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS) )
					writer = new FrameWriterTextCellParallel();
				else
					writer = new FrameWriterTextCell();
				break;
			
			case CSV:
				if( props!=null && !(props instanceof FileFormatPropertiesCSV) )
					throw new DMLRuntimeException("Wrong type of file format properties for CSV writer.");
				if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS) )
					writer = new FrameWriterTextCSVParallel((FileFormatPropertiesCSV)props);
				else
					writer = new FrameWriterTextCSV((FileFormatPropertiesCSV)props);
				break;
			
			case BINARY:
				if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS) )
					writer = new FrameWriterBinaryBlockParallel();
				else
					writer = new FrameWriterBinaryBlock();
				break;

			case PROTO:
				// TODO performance improvement: add parallel reader
				writer = new FrameWriterProto();
				break;
			
			default:
				throw new DMLRuntimeException("Failed to create frame writer for unknown format: " + fmt.toString());
		}
		return writer;
	}
}
