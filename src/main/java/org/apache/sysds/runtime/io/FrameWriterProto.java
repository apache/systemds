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

import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.protobuf.SysdsProtos;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.HDFSTool;

public class FrameWriterProto extends FrameWriter {
	@Override
	public void writeFrameToHDFS(FrameBlock src, String fname, long rlen, long clen) throws IOException {
		// prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);

		// if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS(fname);

		// validity check frame dimensions
		if(src.getNumRows() != rlen || src.getNumColumns() != clen) {
			throw new IOException("Frame dimensions mismatch with metadata: " + src.getNumRows() + "x"
				+ src.getNumColumns() + " vs " + rlen + "x" + clen + ".");
		}

		writeProtoFrameToHDFS(path, job, src, rlen, clen);
	}

	protected void writeProtoFrameToHDFS(Path path, JobConf jobConf, FrameBlock src, long rlen, long clen)
		throws IOException {
		FileSystem fileSystem = IOUtilFunctions.getFileSystem(path, jobConf);
		writeProtoFrameToFile(path, fileSystem, src, 0, (int) rlen);
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fileSystem, path);
	}

	protected void writeProtoFrameToFile(Path path, FileSystem fileSystem, FrameBlock src, int lowerRowBound,
		int upperRowBound) throws IOException {
		// Current Protobuf protocol is based on 32-bit signed arithmetic, meaning potential problems in files of > 2GB.
		// see:
		// https://stackoverflow.com/questions/34128872/google-protobuf-maximum-size#:~:text=Protobuf%20has%20a%20hard%20limit,manually%20if%20you%20need%20to.
		OutputStream outputStream = fileSystem.create(path, true);
		SysdsProtos.Frame.Builder frameBuilder = SysdsProtos.Frame.newBuilder();
		try {
			Iterator<String[]> stringRowIterator = src.getStringRowIterator(lowerRowBound, upperRowBound);
			while(stringRowIterator.hasNext()) {
				String[] row = stringRowIterator.next();
				frameBuilder.addRowsBuilder().addAllColumnData(Arrays.asList(row));
			}
			frameBuilder.build().writeTo(outputStream);
		}
		finally {
			IOUtilFunctions.closeSilently(outputStream);
		}
	}
}
