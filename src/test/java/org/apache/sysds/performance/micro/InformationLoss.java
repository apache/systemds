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

package org.apache.sysds.performance.micro;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.performance.generators.FrameFile;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.WriterTextCSV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.transform.decode.ColumnDecoder;
import org.apache.sysds.runtime.transform.decode.ColumnDecoderFactory;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;

/**
 * evaluate the different loss in accuracy on different number of distinct values on transform specifications.
 */
public class InformationLoss {

	public static void main(String[] args) throws Exception {

		String frame_path = args[0];
		String binningTechnique = args[1];

		writeRandomMatrix(frame_path);

		final Pair<FrameBlock, MatrixBlock> p = readFrame(frame_path);
		final FrameBlock f = p.getKey();
		final MatrixBlock org = p.getValue();

		System.gc(); // indicate to do garbage collection here.

		for(int i = 1; i < 20; i++) {
			String spec = generateSpec(i, f.getNumColumns(), binningTechnique);
			System.out.print(i + ",");
			calculateLoss(f, org, spec);
		}

		for(int i = 20; i < 200; i += 10) {
			String spec = generateSpec(i, f.getNumColumns(), binningTechnique);
			System.out.print(i + ",");
			calculateLoss(f, org, spec);
		}

		for(int i = 200; i <= 2000; i += 100) {
			String spec = generateSpec(i, f.getNumColumns(), binningTechnique);
			System.out.print(i + ",");
			calculateLoss(f, org, spec);
		}

	}

	private static Pair<FrameBlock, MatrixBlock> readFrame(String path) throws Exception {
		FrameBlock f = FrameFile.create(path).take();
		f = f.applySchema(f.detectSchema(16), 16); // apply scheme

		MatrixBlock org = DataConverter.convertToMatrixBlock(f);
		f = null;
		System.gc(); // cleanup original frame.

		// normalize org.
		org = org//
			.replaceOperations(null, Double.NaN, 0)//
			.replaceOperations(null, Double.POSITIVE_INFINITY, 0).replaceOperations(null, Double.NEGATIVE_INFINITY, 0);

		Pair<MatrixBlock, MatrixBlock> mm = getMinMax(org);

		// normalize org to 0-1 range
		org = org //
			.binaryOperations(new BinaryOperator(Minus.getMinusFnObject()), mm.getKey())
			.binaryOperations(new BinaryOperator(Divide.getDivideFnObject()),
				mm.getValue().binaryOperations(new BinaryOperator(Minus.getMinusFnObject()), mm.getKey()))
			.replaceOperations(null, Double.NaN, 0);

		f = DataConverter.convertToFrameBlock(org, 16);

		return new Pair<>(f, org);
	}

	private static Pair<MatrixBlock, MatrixBlock> getMinMax(final MatrixBlock org) throws Exception {
		ExecutorService pool = CommonThreadPool.get(16);
		try{

			Future<MatrixBlock> minF = pool.submit(() -> org.colMin(16));
			Future<MatrixBlock> maxF = pool.submit(() -> org.colMax(16));
	
			MatrixBlock min = minF.get();
			MatrixBlock max = maxF.get();
	
			return new Pair<>(min, max);
		}
		finally{
			pool.shutdown();
		}

	}

	private static void writeRandomMatrix(String path) throws IOException {
		if(!new File(path).exists()) {
			MatrixWriter w = new WriterTextCSV(new FileFormatPropertiesCSV(false, ",", false));
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 10, 0, 1, 0.5, 23);
			w.writeMatrixToHDFS(mb, path, mb.getNumRows(), mb.getNumColumns(), 1000, mb.getNonZeros(), false);
		}
	}

	private static FrameBlock encodeAndDecode(FrameBlock f, MatrixBlock org, String spec) {

		MultiColumnEncoder encoder = //
			EncoderFactory.createEncoder(spec, f.getColumnNames(), f.getNumColumns(), null);
		MatrixBlock binned = encoder.encode(f, 16, true);
		ColumnDecoder d = ColumnDecoderFactory.createDecoder(spec, f.getColumnNames(false), f.getSchema(), encoder.getMetaData(null),
		//Decoder d = DecoderFactory.createDecoder(spec, f.getColumnNames(false), f.getSchema(), encoder.getMetaData(null),
			binned.getNumColumns());
		FrameBlock dr = new FrameBlock(f.getSchema());
		d.columnDecode(binned, dr, 16);
		return dr;
	}

	private static MatrixBlock delta(FrameBlock f, MatrixBlock org, String spec) {
		return DataConverter//
			.convertToMatrixBlock(encodeAndDecode(f, org, spec))
			.binaryOperations(new BinaryOperator(Minus.getMinusFnObject(), 16), org)
			// .binaryOperations(new BinaryOperator(Divide.getDivideFnObject(), 16), org)
			.unaryOperations(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.ABS), 16), null)
			.replaceOperations(null, Double.NaN, 0);

	}

	private static void calculateLoss(FrameBlock f, MatrixBlock org, String spec) throws Exception {

		final MatrixBlock delta = delta(f, org, spec);
		ExecutorService pool = CommonThreadPool.get(16);
		try{

			Future<Double> minF = pool.submit(() -> delta.min(16));
			Future<Double> maxF = pool.submit(() -> delta.max(16).get(0, 0));
			Future<Double> meanF = pool
				.submit(() -> delta.sum(16).get(0, 0) / (delta.getNumRows() * delta.getNumColumns()));
	
			double min = minF.get();
			double max = maxF.get();
			double mean = meanF.get();

			System.out.println(String.format("%e, %e, %e", min, max, mean));
		}
		finally{
			pool.shutdown();
		}

	}

	private static String generateSpec(int bins, int cols, String technique) {
		StringBuilder sb = new StringBuilder();
		sb.append("{\"ids\":true,\"bin\":[");
		for(int i = 0; i < cols; i++) {
			sb.append(String.format("{\"id\":%d,\"method\":\"%s\",\"numbins\":%d}", i + 1, technique, bins));
			if(i + 1 < cols)
				sb.append(',');
		}
		sb.append("]}");
		return sb.toString();

	}

}
