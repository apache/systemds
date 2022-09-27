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

package org.apache.sysds.test.component.compress.io;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.File;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.compress.io.CompressedWriteBlock;
import org.apache.sysds.runtime.compress.io.WriterCompressed;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.io.InputOutputInfo;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.Test;

import scala.Tuple2;

public class IOSpark {

	protected static final Log LOG = LogFactory.getLog(IOSpark.class.getName());

	final static String nameBeginning = "src/test/java/org/apache/sysds/test/component/compress/io/files"
		+ IOSpark.class.getSimpleName() + "/";

	@AfterClass
	public static void cleanup() {
		IOTestUtils.deleteDirectory(new File(nameBeginning));
	}

	private static String getName() {
		return IOTestUtils.getName(nameBeginning);
	}

	@Test
	public void readEmpty() {
		String n = getName();
		IOEmpty.write(n, 1000, 102, 100);
		verifySum(read(n), 0, 0);
	}

	@Test
	public void readSPContextEmpty() {
		readRDDThroughSparkExecutionContext(new MatrixBlock(100, 100, 0.0), 40);
	}

	@Test
	public void readSPContextCompressable() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(120, 140, 1, 3, 1.0, 2514));
		readRDDThroughSparkExecutionContext(mb, 100);
	}

	@Test
	public void readSPContextNotCompressable() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(120, 140, 1, 3, 1.0, 2514);
		readRDDThroughSparkExecutionContext(mb, 100);
	}

	@Test
	public void readMultiBlockCols() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 51, 1, 3, 1.0, 2514));
		readWrite(mb);
	}

	@Test
	public void readMultiBlockRows() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(120, 39, 1, 3, 1.0, 2514));
		readWrite(mb);
	}

	@Test
	public void readMultiBlockRowsAndCols() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(120, 122, 1, 3, 1.0, 2514));
		readWrite(mb);
	}

	@Test
	public void readMultiBlockRowsAndColsIncompressable() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(120, 122, 1, 1000, 1.0, 2514));
		readWrite(mb);
	}

	private void readWrite(MatrixBlock mb) {
		double sum = mb.sum();
		String n = getName();
		try {
			WriterCompressed.writeCompressedMatrixToHDFS(mb, n, 50);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		verifySum(read(n), sum, 0.0001);
	}

	private void verifySum(List<Tuple2<MatrixIndexes, MatrixBlock>> c, double val, double tol) {
		double sum = 0.0;
		for(Tuple2<MatrixIndexes, MatrixBlock> b : c)
			sum += b._2().sum();
		assertEquals(val, sum, tol);
	}

	private List<Tuple2<MatrixIndexes, MatrixBlock>> read(String n) {
		JavaPairRDD<MatrixIndexes, MatrixBlock> m = getRDD(n).mapValues(x -> x.get());
		return m.collect();

	}

	@SuppressWarnings({"unchecked"})
	private JavaPairRDD<MatrixIndexes, CompressedWriteBlock> getRDD(String path) {
		InputOutputInfo inf = InputOutputInfo.CompressedInputOutputInfo;
		JavaSparkContext sc = SparkExecutionContext.getSparkContextStatic();
		return (JavaPairRDD<MatrixIndexes, CompressedWriteBlock>) sc.hadoopFile(path, inf.inputFormatClass, inf.keyClass,
			inf.valueClass);
	}

	@SuppressWarnings({"unchecked"})
	public void readRDDThroughSparkExecutionContext(MatrixBlock mb, int blen) {
		try {

			String n = getName();

			WriterCompressed.writeCompressedMatrixToHDFS(mb, n, blen);

			SparkExecutionContext ec = ExecutionContextFactory.createSparkExecutionContext();

			MatrixObject obj = new MatrixObject(ValueType.FP64, n);
			FileFormat fmt = FileFormat.COMPRESSED;

			JavaPairRDD<MatrixIndexes, MatrixBlock> m = (JavaPairRDD<MatrixIndexes, MatrixBlock>) ec
				.getRDDHandleForMatrixObject(obj, fmt);

			verifySum(m.collect(), mb.sum(), 0.0001);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

}
