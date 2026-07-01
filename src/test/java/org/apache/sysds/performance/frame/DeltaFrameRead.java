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

package org.apache.sysds.performance.frame;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

import org.apache.commons.io.FileUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.performance.compression.APerfTest;
import org.apache.sysds.performance.generators.ConstFrame;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderDelta;
import org.apache.sysds.runtime.io.FrameReaderDeltaParallel;
import org.apache.sysds.runtime.io.FrameWriterDelta;
import org.apache.sysds.test.TestUtils;

/**
 * Reads the SAME native Delta frame table from disk repeatedly and reports read
 * throughput. The table is written to a temporary directory ONCE as (untimed)
 * setup; every timed repetition re-opens the latest snapshot and materializes a
 * fresh {@link FrameBlock}, so the numbers reflect the read path only (parquet
 * decode + column materialization), not the write.
 *
 * <p>This is the target for an async-profiler run: launch the perf jar under the
 * profiler agent and this loop provides a long, steady-state read workload to
 * sample. See {@code src/test/java/org/apache/sysds/performance/README.md} and
 * the {@code delta-async-profiler} cursor rule.</p>
 *
 * <p>Dispatched from {@link org.apache.sysds.performance.Main} (program id 18).</p>
 */
public class DeltaFrameRead extends APerfTest<Long, FrameBlock> {

	//the Delta reader derives schema/names from the table metadata, so the values
	//passed here are placeholders (a single detect column) and are ignored.
	private static final ValueType[] DETECT_SCHEMA = new ValueType[] {ValueType.STRING};
	private static final String[] DETECT_NAMES = new String[] {"x"};

	private final int k;
	private final String mode;
	private final long targetFileSize; //<=0 -> adaptive default sizing

	private String tablePath;
	private Path tableDir;
	private long inMemSize;
	private long files;

	public DeltaFrameRead(int N, IGenerate<FrameBlock> gen, int k, String mode, long targetFileSize) {
		super(N, gen);
		this.k = k;
		this.mode = mode;
		this.targetFileSize = targetFileSize;
	}

	public void run() throws Exception {
		try {
			setup();
			System.out.println(this);
			System.out.printf("table: %s%n", tablePath);
			System.out.printf("layout: files=%d, in-memory=%.1f MB, target=%s%n",
				files, inMemSize / 1048576.0,
				targetFileSize > 0 ? (targetFileSize / 1048576) + "MB(fixed)" : "adaptive");

			if( mode.equals("serial") || mode.equals("both") )
				execute(() -> readSerial(), "Delta read serial");
			if( mode.equals("parallel") || mode.equals("both") )
				execute(() -> readParallel(), "Delta read parallel(k=" + k + ")");
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			if( tableDir != null )
				FileUtils.deleteQuietly(tableDir.toFile());
		}
	}

	/** Untimed: materialize the source frame and write it to a temp Delta table once. */
	private void setup() throws Exception {
		FrameBlock fb = gen.take();
		inMemSize = fb.getInMemorySize();

		DMLConfig c = new DMLConfig();
		if( targetFileSize > 0 ) {
			c.setTextValue(DMLConfig.DELTA_WRITER_ADAPTIVE_FILE_SIZE, "false");
			c.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(targetFileSize));
		}
		ConfigurationManager.setLocalConfig(c);

		tableDir = Files.createTempDirectory("sysds_delta_frame_read_");
		tablePath = new File(tableDir.toFile(), "table").getAbsolutePath();
		new FrameWriterDelta().writeFrameToHDFS(fb, tablePath, fb.getNumRows(), fb.getNumColumns());
		files = countParquet(tablePath);
	}

	private void readSerial() {
		try {
			FrameBlock fb = new FrameReaderDelta()
				.readFrameFromHDFS(tablePath, DETECT_SCHEMA, DETECT_NAMES, -1, -1);
			ret.add(fb.getInMemorySize());
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	private void readParallel() {
		try {
			FrameBlock fb = new FrameReaderDeltaParallel()
				.readFrameFromHDFS(tablePath, DETECT_SCHEMA, DETECT_NAMES, -1, -1);
			ret.add(fb.getInMemorySize());
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	private static long countParquet(String tablePath) throws Exception {
		try( Stream<Path> s = Files.walk(new File(tablePath).toPath()) ) {
			return s.filter(p -> p.toString().endsWith(".parquet")).count();
		}
	}

	@Override
	protected String makeResString() {
		throw new RuntimeException("Do not call");
	}

	@Override
	protected String makeResString(double[] times) {
		double meanMs = trimmedMean(times);
		double mbPerSec = (inMemSize / 1048576.0) / (meanMs / 1000.0);
		return String.format("%8.1f MB/s", mbPerSec);
	}

	/** 5%-trimmed mean, matching the trimming used by the framework statistics. */
	private static double trimmedMean(double[] times) {
		double[] v = times.clone();
		java.util.Arrays.sort(v);
		int remove = (int) Math.floor(v.length * 0.05);
		double total = 0;
		int el = v.length - remove * 2;
		for( int i = remove; i < v.length - remove; i++ )
			total += v[i];
		return total / Math.max(el, 1);
	}

	@Override
	public String toString() {
		return super.toString() + " mode: " + mode + ", threads: " + k;
	}

	/** Build a representative mixed-schema frame (string + numeric columns). */
	public static IGenerate<FrameBlock> mixedFrame(int rows, long seed) {
		ValueType[] schema = new ValueType[] {ValueType.STRING, ValueType.INT64, ValueType.FP64,
			ValueType.BOOLEAN, ValueType.INT32, ValueType.FP32};
		return new ConstFrame(TestUtils.generateRandomFrameBlock(rows, schema, seed));
	}
}
