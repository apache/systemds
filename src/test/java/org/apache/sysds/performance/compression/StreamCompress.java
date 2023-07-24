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

package org.apache.sysds.performance.compression;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class StreamCompress extends APerfTest<Double, MatrixBlock> {

	public StreamCompress(int N, IGenerate<MatrixBlock> gen) {
		super(N, gen);
	}

	public void run() throws Exception, InterruptedException, IOException {
		System.out.println("Running Steam Compression Test");
		System.out.println(this);

		warmup(() -> sumTask(), 10);
		execute(() -> blockSizeTask(), "In Memory Block Size");
		execute(() -> writeSteam(), "Write Blocks Stream");
		execute(() -> writeSteamDeflaterOutputStreamDef(), "Write Stream Deflate");
		execute(() -> writeSteamDeflaterOutputStreamSpeed(), "Write Stream Deflate Speedy");
		execute(() -> compressTask(), "In Memory Compress Individual (CI)");
		execute(() -> writeStreamCompressTask(), "Write CI Stream");
		execute(() -> writeStreamCompressDeflaterOutputStreamTask(), "Write CI Deflate Stream");
		execute(() -> writeStreamCompressDeflaterOutputStreamTaskSpeedy(), "Write CI Deflate Stream Speedy");

	}

	@Override
	protected String makeResString() {
		Double avgRes = ret.stream().mapToDouble(a -> a).average().getAsDouble();
		return String.format("%10.2f", avgRes);
	}

	private void sumTask() {
		ret.add(gen.take().sum());
	}

	private void blockSizeTask() {
		ret.add((double) gen.take().getInMemorySize());
	}

	private void compressTask() {

		MatrixBlock mb = CompressedMatrixBlockFactory.compress(gen.take()).getLeft();
		ret.add((double) mb.getInMemorySize());

	}

	private void writeStreamCompressTask() {

		MatrixBlock mb = CompressedMatrixBlockFactory.compress(gen.take()).getLeft();
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream fos = new DataOutputStream(bos);
		try {
			mb.write(fos);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		ret.add((double) bos.size());

	}

	private void writeStreamCompressDeflaterOutputStreamTask() {

		MatrixBlock mb = CompressedMatrixBlockFactory.compress(gen.take()).getLeft();
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DeflaterOutputStream decorator = new DeflaterOutputStream(bos);
		DataOutputStream fos = new DataOutputStream(decorator);
		try {
			mb.write(fos);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		ret.add((double) bos.size());

	}

	private void writeStreamCompressDeflaterOutputStreamTaskSpeedy() {

		MatrixBlock mb = CompressedMatrixBlockFactory.compress(gen.take()).getLeft();
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DeflaterOutputStream decorator = new DeflaterOutputStream(bos, new Deflater(Deflater.BEST_SPEED));
		DataOutputStream fos = new DataOutputStream(decorator);
		try {
			mb.write(fos);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		ret.add((double) bos.size());

	}

	private void writeSteam() {

		MatrixBlock mb = gen.take();
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream fos = new DataOutputStream(bos);
		try {
			mb.write(fos);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		ret.add((double) bos.size());

	}

	private void writeSteamDeflaterOutputStreamDef() {

		MatrixBlock mb = gen.take();
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DeflaterOutputStream decorator = new DeflaterOutputStream(bos);
		DataOutputStream fos = new DataOutputStream(decorator);
		try {
			mb.write(fos);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		ret.add((double) bos.size());

	}

	private void writeSteamDeflaterOutputStreamSpeed() {

		MatrixBlock mb = gen.take();
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DeflaterOutputStream decorator = new DeflaterOutputStream(bos, new Deflater(Deflater.BEST_SPEED));
		DataOutputStream fos = new DataOutputStream(decorator);
		try {
			mb.write(fos);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		ret.add((double) bos.size());

	}

}
