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

package org.apache.sysds.runtime.frame.data.compress;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.DDCArray;

public class CompressedFrameBlockFactory {

	private static final Log LOG = LogFactory.getLog(CompressedFrameBlockFactory.class.getName());

	private final FrameBlock in;
	private final FrameCompressionSettings cs;
	private final ArrayCompressionStatistics[] stats;
	private final Array<?>[] compressedColumns;

	private CompressedFrameBlockFactory(FrameBlock fb, FrameCompressionSettings cs) {
		this.in = fb;
		this.cs = cs;
		this.stats = new ArrayCompressionStatistics[in.getNumColumns()];
		this.compressedColumns = new Array<?>[in.getNumColumns()];
	}

	public static FrameBlock compress(FrameBlock fb) {
		FrameCompressionSettings cs = new FrameCompressionSettingsBuilder().create();
		return compress(fb, cs);
	}

	public static FrameBlock compress(FrameBlock fb, int k, WTreeRoot root) {
		FrameCompressionSettings cs = new FrameCompressionSettingsBuilder()//
			.threads(k).wTreeRoot(root).create();
		return compress(fb, cs);
	}

	public static FrameBlock compress(FrameBlock fb, FrameCompressionSettingsBuilder csb) {
		return compress(fb, csb.create());
	}

	public static FrameBlock compress(FrameBlock fb, FrameCompressionSettings cs) {
		return new CompressedFrameBlockFactory(fb, cs).compressFrame();
	}

	private FrameBlock compressFrame() {
		extractStatistics();
		logStatistics();
		encodeColumns();
		final FrameBlock ret = new FrameBlock(compressedColumns, in.getColumnNames(false));
		logRet(ret);
		return ret;
	}

	private void extractStatistics() {
		final int nSamples = Math.min(in.getNumRows(), (int) Math.ceil(in.getNumRows() * cs.sampleRatio));
		for(int i = 0; i < stats.length; i++) {
			stats[i] = in.getColumn(i).statistics(nSamples);
		}
	}

	private void encodeColumns() {
		for(int i = 0; i < compressedColumns.length; i++) {
			if(stats[i] != null) {
				// commented out because no other encodings are supported yet
				// switch(stats[i].bestType) {
				// case DDC:
				compressedColumns[i] = DDCArray.compressToDDC(in.getColumn(i));
				// break;
				// default:
				// compressedColumns[i] = in.getColumn(i);
				// break;
				// }
			}
			else
				compressedColumns[i] = in.getColumn(i);
		}
	}

	private void logStatistics() {
		if(LOG.isDebugEnabled()) {
			for(int i = 0; i < compressedColumns.length; i++) {
				if(stats[i] != null)
					LOG.debug(stats[i]);
				else
					LOG.debug("no Comp col: " + i);
			}
		}
	}

	private void logRet(FrameBlock ret) {
		if(LOG.isDebugEnabled()) {
			final long before = in.getInMemorySize();
			final long after = ret.getInMemorySize();
			LOG.debug(String.format("Uncompressed Size: %15d", before));
			LOG.debug(String.format("compressed Size:   %15d", after));
			LOG.debug(String.format("ratio:             %15.3f", (double) before / (double) after));
		}
	}

}
