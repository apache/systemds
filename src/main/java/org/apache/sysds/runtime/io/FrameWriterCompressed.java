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

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameLibCompress;

public class FrameWriterCompressed extends FrameWriterBinaryBlockParallel {

	private final boolean parallel;

	public FrameWriterCompressed(boolean parallel) {
		this.parallel = parallel;
	}

	@Override
	protected void writeBinaryBlockFrameToHDFS(Path path, JobConf job, FrameBlock src, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		int k = parallel ? OptimizerUtils.getParallelBinaryWriteParallelism() : 1;
		FrameBlock compressed = FrameLibCompress.compress(src, k).getLeft();
		super.writeBinaryBlockFrameToHDFS(path, job, compressed, rlen, clen);
	}

}
