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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.compress.io.CompressUnwrap;
import org.apache.sysds.runtime.compress.io.CompressedWriteBlock;
import org.apache.sysds.runtime.compress.io.WriterCompressed;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysds.runtime.io.InputOutputInfo;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
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
		IOCompressionTestUtils.deleteDirectory(new File(nameBeginning));
	}

	private static String getName() {
		return IOCompressionTestUtils.getName(nameBeginning);
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

	@Test
	public void writeSparkReadCPMultiColBlock() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 124, 1, 3, 1.0, 2514));
		testWriteSparkReadCP(mb, 100, 100);
	}

	@Test
	public void writeSparkReadCPMultiRowBlock() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1322, 33, 1, 3, 1.0, 2514));
		testWriteSparkReadCP(mb, 100, 100);
	}

	@Test
	public void writeSparkReadCPSingleBlock() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 99, 1, 3, 1.0, 33));
		testWriteSparkReadCP(mb, 100, 100);
	}

	@Test
	public void writeSparkReadCPMultiBlock() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(580, 244, 1, 3, 1.0, 33));
		testWriteSparkReadCP(mb, 100, 100);
	}

	@Test
	public void writeSparkReadCPMultiColBlockReblockUp() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 124, 1, 3, 1.0, 2514));
		testWriteSparkReadCP(mb, 100, 150);
	}

	@Test
	public void writeSparkReadCPMultiRowBlockReblockUp() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1322, 33, 1, 3, 1.0, 2514));
		testWriteSparkReadCP(mb, 100, 150);
	}

	@Test
	public void writeSparkReadCPSingleBlockReblockUp() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 99, 1, 3, 1.0, 33));
		testWriteSparkReadCP(mb, 100, 150);
	}

	@Test
	public void writeSparkReadCPMultiBlockReblockUp() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(580, 244, 1, 3, 1.0, 33));
		testWriteSparkReadCP(mb, 100, 150);
	}

	@Test
	public void writeSparkReadCPMultiColBlockReblockDown() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 124, 1, 3, 1.0, 2514));
		testWriteSparkReadCP(mb, 100, 80);
	}

	@Test
	public void writeSparkReadCPMultiRowBlockReblockDown() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1322, 33, 1, 3, 1.0, 2514));
		testWriteSparkReadCP(mb, 100, 80);
	}

	@Test
	public void writeSparkReadCPSingleBlockReblockDown() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 99, 1, 3, 1.0, 33));
		testWriteSparkReadCP(mb, 100, 80);
	}

	@Test
	public void writeSparkReadCPMultiBlockReblockDown() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(580, 244, 1, 3, 1.0, 33));
		testWriteSparkReadCP(mb, 100, 80);
	}

	private void testWriteSparkReadCP(MatrixBlock mb, int blen1, int blen2) throws Exception {

		String f1 = getName();
		WriterCompressed.writeCompressedMatrixToHDFS(mb, f1, blen1);

		// Make sure the first file is written
		File f = new File(f1);
		assertTrue(f.isFile() || f.isDirectory());
		// Read in again as RDD
		JavaPairRDD<MatrixIndexes, MatrixBlock> m = getRDD(f1);
		String f2 = getName(); // get new name for writing RDD.

		// Write RDD to disk
		if(blen1 != blen2) {
			DataCharacteristics mc = new MatrixCharacteristics(mb.getNumRows(), mb.getNumColumns(), blen1);
			WriterCompressed.writeRDDToHDFS(m, f2, blen2, mc);
		}
		else
			WriterCompressed.writeRDDToHDFS(m, f2);

		// Read locally the spark block written.
		MatrixBlock mbr = IOCompressionTestUtils.read(f2);
		IOCompressionTestUtils.verifyEquivalence(mb, mbr);
	}

	@Test
	public void testReblock_up() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 50, 1, 3, 1.0, 2514));
		testReblock(mb, 25, 50);
	}

	@Test
	public void testReblock_up_2() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 50, 1, 3, 1.0, 2514));
		testReblock(mb, 25, 55);
	}

	@Test
	public void testReblock_up_3() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(165, 110, 1, 3, 1.0, 2514));
		testReblock(mb, 25, 55);
	}

	@Test
	public void testReblock_up_4() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(165, 110, 1, 3, 1.0, 2514));
		testReblock(mb, 25, 100);
	}

	@Test
	public void testReblock_up_5() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(230, 401, 1, 3, 1.0, 2514));
		testReblock(mb, 25, 100);
	}

	@Test
	public void testReblock_down() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 50, 1, 3, 1.0, 2514));
		testReblock(mb, 50, 25);
	}

	@Test
	public void testReblock_down_2() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 50, 1, 3, 1.0, 2514));
		testReblock(mb, 55, 25);
	}

	@Test
	public void testReblock_down_3() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(165, 110, 1, 3, 1.0, 2514));
		testReblock(mb, 55, 25);
	}

	@Test
	public void testReblock_down_4() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(165, 110, 1, 3, 1.0, 2514));
		testReblock(mb, 100, 25);
	}

	@Test
	public void testReblock_down_5() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(230, 401, 1, 3, 1.0, 2514));
		testReblock(mb, 100, 25);
	}

	private void testReblock(MatrixBlock mb, int blen1, int blen2) throws Exception {
		String f1 = getName();
		WriterCompressed.writeCompressedMatrixToHDFS(mb, f1, blen1);

		// Read in again as RDD
		JavaPairRDD<MatrixIndexes, MatrixBlock> m = getRDD(f1); // Our starting point

		int nBlocksExpected = (1 + (mb.getNumColumns() - 1) / blen1) * (1 + (mb.getNumRows() - 1) / blen1);
		int nBlocksActual = m.collect().size();
		assertEquals("Expected same number of blocks ", nBlocksExpected, nBlocksActual);

		DataCharacteristics mc = new MatrixCharacteristics(mb.getNumRows(), mb.getNumColumns(), blen1);
		JavaPairRDD<MatrixIndexes, MatrixBlock> m2 = reblock(m, mc, blen2);
		int nBlocksExpected2 = (1 + (mb.getNumColumns() - 1) / blen2) * (1 + (mb.getNumRows() - 1) / blen2);
		int nBlocksActual2 = m2.collect().size();
		assertEquals("Expected same number of blocks on re-blocked", nBlocksExpected2, nBlocksActual2);

		double val = mb.sum();
		verifySum(m, val, 0.0000001);
		verifySum(m2, val, 0.0000001);

	}

	private static JavaPairRDD<MatrixIndexes, MatrixBlock> reblock(JavaPairRDD<MatrixIndexes, MatrixBlock> in,
		DataCharacteristics mc, int blen) {
		final DataCharacteristics outC = new MatrixCharacteristics(mc).setBlocksize(blen);
		return RDDConverterUtils.binaryBlockToBinaryBlock(in, mc, outC);
	}

	private List<Tuple2<MatrixIndexes, MatrixBlock>> read(String n) {
		return getRDD(n).collect();
	}

	@SuppressWarnings({"unchecked"})
	private JavaPairRDD<MatrixIndexes, MatrixBlock> getRDD(String path) {
		InputOutputInfo inf = InputOutputInfo.CompressedInputOutputInfo;
		JavaSparkContext sc = SparkExecutionContext.getSparkContextStatic();
		return ((JavaPairRDD<MatrixIndexes, CompressedWriteBlock>) sc.hadoopFile(path, inf.inputFormatClass, inf.keyClass,
			inf.valueClass)).mapValues(new CompressUnwrap());
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
			List<Tuple2<MatrixIndexes, MatrixBlock>> c = m.collect();
			verifySum(c, mb.sum(), 0.0001);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
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

	private void verifySum(JavaPairRDD<MatrixIndexes, MatrixBlock> m, double val, double tol) {
		verifySum(m.collect(), val, tol);
	}

	private void verifySum(List<Tuple2<MatrixIndexes, MatrixBlock>> c, double val, double tol) {
		double sum = 0.0;
		for(Tuple2<MatrixIndexes, MatrixBlock> b : c)
			sum += b._2().sum();
		assertEquals(val, sum, tol);
	}
}
