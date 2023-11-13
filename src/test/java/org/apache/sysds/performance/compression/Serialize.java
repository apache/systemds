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

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.ArrayList;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.colgroup.scheme.CompressionScheme;
import org.apache.sysds.runtime.compress.io.ReaderCompressed;
import org.apache.sysds.runtime.compress.io.WriterCompressed;
import org.apache.sysds.runtime.compress.lib.CLALibScheme;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.ReaderBinaryBlock;
import org.apache.sysds.runtime.io.ReaderBinaryBlockParallel;
import org.apache.sysds.runtime.io.WriterBinaryBlock;
import org.apache.sysds.runtime.io.WriterBinaryBlockParallel;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class Serialize extends APerfTest<Serialize.InOut, MatrixBlock> {

	private final String file;
	private final int k;
	private final String codec;

	public Serialize(int N, IGenerate<MatrixBlock> gen) {
		super(N, gen);
		file = "./tmp/perf-tmp.bin";
		k = 1;
		codec = "none";
	}

	public Serialize(int N, IGenerate<MatrixBlock> gen, int k) {
		super(N, gen);
		file = "./tmp/perf-tmp.bin";
		this.k = k;
		codec = "none";
	}

	public Serialize(int N, IGenerate<MatrixBlock> gen, int k, String file) {
		super(N, gen);
		this.file = file;
		this.k = k;
		codec = "none";

	}

	public Serialize(int N, IGenerate<MatrixBlock> gen, int k, String file, String codec) {
		super(N, gen);
		this.file = file == null ? "tmp/perf-tmp.bin" : file;
		this.k = k;
		this.codec = codec;
	}

	public void run() throws Exception, InterruptedException {
		CompressedMatrixBlock.debug = true;
		CompressedMatrixBlock.debug = false;
		System.out.println(this);
		File directory = new File(file).getParentFile();
		if(!directory.exists()) {
			directory.mkdir();
		}

		if(k == 1) {
			ConfigurationManager.getCompilerConfig().set(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS, false);
		}
		
		ConfigurationManager.getDMLConfig().setTextValue(DMLConfig.IO_COMPRESSION_CODEC, codec);
		System.out.println(ConfigurationManager.getDMLConfig().getTextValue(DMLConfig.IO_COMPRESSION_CODEC));
		warmup(() -> sumTask(k), N);


		// execute(() -> writeUncompressed(k), "Serialize");
		// execute(() -> diskUncompressed(k), "CustomDisk");
		// execute(() -> compressTask(k), "Compress Normal");
		// execute(() -> writeCompressTask(k), "Compress Normal Serialize");
		// execute(() -> diskCompressTask(k), "Compress Normal CustomDisk");
		// execute(() -> updateAndApplySchemeFused(sch2, k), "Update&Apply Scheme Fused");
		// execute(() -> writeUpdateAndApplySchemeFused(sch2, k), "Update&Apply Scheme Fused Serialize");
		// execute(() -> diskUpdateAndApplySchemeFused(sch2, k), "Update&Apply Scheme Fused Disk");
		
		execute(() -> standardIO(k), () -> setFileSize(), () -> cleanup(), "StandardDisk");
		execute(() -> standardCompressedIO(k), () -> setFileSize(), () -> cleanup(), "Compress StandardIO");
		final CompressionScheme sch2 = CLALibScheme.getScheme(getC());
		execute(() -> standardCompressedIOUpdateAndApply(sch2, k), () -> setFileSize(), () -> cleanup(),
			"Update&Apply Standard IO");

		// write the input file to disk.
		standardIO(k);
		execute(() -> standardIORead(k), "StandardRead");
		cleanup();
		// write compressed input file to disk
		standardCompressedIOUpdateAndApply(sch2, k);
		// standardCompressedIO( k);
		execute(() -> standardCompressedRead(k), "StandardCompressedRead");
	}

	public void run(int i) throws Exception, InterruptedException {
		warmup(() -> sumTask(k), N);
		if(k == 1) {
			ConfigurationManager.getDMLConfig().setTextValue(DMLConfig.CP_PARALLEL_IO, "false");
			ConfigurationManager.getCompilerConfig().set(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS, false);
		}

		final CompressionScheme sch = (i == 8 || i == 9 || i == 10 || i == 11) ? CLALibScheme.getScheme(getC()) : null;
		cleanup();
		switch(i) {
			case 1:
				execute(() -> writeUncompressed(k), "Serialize");
				break;
			case 2:
				execute(() -> diskUncompressed(k), "CustomDisk");
				break;
			case 3:
				execute(() -> standardIO(k), () -> setFileSize(), () -> cleanup(), "StandardDisk");
				break;
			case 4:
				execute(() -> compressTask(k), "Compress Normal");
				break;
			case 5:
				execute(() -> writeCompressTask(k), "Compress Normal Serialize");
				break;
			case 6:
				execute(() -> diskCompressTask(k), "Compress Normal CustomDisk");
				break;
			case 7:
				execute(() -> standardCompressedIO(k), () -> setFileSize(), () -> cleanup(), "Compress StandardIO");
				break;
			case 8:
				execute(() -> updateAndApplySchemeFused(sch, k), "Update&Apply Scheme Fused");
				break;
			case 9:
				execute(() -> writeUpdateAndApplySchemeFused(sch, k), "Update&Apply Scheme Fused Serialize");
				break;
			case 10:
				execute(() -> diskUpdateAndApplySchemeFused(sch, k), "Update&Apply Scheme Fused Disk");
				break;
			case 11:
				execute(() -> standardCompressedIOUpdateAndApply(sch, k), () -> setFileSize(), () -> cleanup(),
					"Update&Apply Standard IO");
				break;
		}
		// cleanup();
	}

	private void writeUncompressed(int k) {
		MatrixBlock mb = gen.take();
		Sink o = serialize(mb);
		ret.add(new InOut(mb.getInMemorySize(), o.size()));
	}

	private void diskUncompressed(int k) {
		MatrixBlock mb = gen.take();
		Disk o = serializeD(mb);
		ret.add(new InOut(mb.getInMemorySize(), o.size()));
	}

	private void standardIO(int k) {
		try {

			MatrixWriter w = (k == 1) ? new WriterBinaryBlock(-1) : new WriterBinaryBlockParallel(1);
			MatrixBlock mb = gen.take();
			w.writeMatrixToHDFS(mb, file, mb.getNumRows(), mb.getNumColumns(), 1000, mb.getNonZeros(), false);
			ret.add(new InOut(mb.getInMemorySize(), -1));
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	private void standardIORead(int k) {
		try {
			MatrixBlock mb = gen.take();
			MatrixReader r = (k == 1) ? new ReaderBinaryBlock(false) : new ReaderBinaryBlockParallel(false);
			MatrixBlock mbr = r.readMatrixFromHDFS(file, mb.getNumRows(), mb.getNumColumns(), ConfigurationManager.getBlocksize(), mb.getNonZeros());

			ret.add(new InOut(mb.getInMemorySize(),mbr.getInMemorySize()));
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	private void compressTask(int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock cmb = CompressedMatrixBlockFactory.compress(mb, k).getLeft();
		long out = cmb.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void writeCompressTask(int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock cmb = CompressedMatrixBlockFactory.compress(mb, k).getLeft();
		Sink o = serialize(cmb);
		ret.add(new InOut(in, o.size()));
	}

	private void diskCompressTask(int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock cmb = CompressedMatrixBlockFactory.compress(mb, k).getLeft();
		Disk o = serializeD(cmb);
		ret.add(new InOut(in, o.size()));
	}

	private void standardCompressedIO(int k) {
		try {
			// MatrixWriter w = new WriterBinaryBlockParallel(1);
			MatrixBlock mb = gen.take();
			WriterCompressed.writeCompressedMatrixToHDFS(mb, file);
			ret.add(new InOut(mb.getInMemorySize(), -1));
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	private void standardCompressedRead(int k) {
		try {
			MatrixBlock mb = gen.take();
			ReaderCompressed r = new ReaderCompressed(k);
			MatrixBlock mbr = r.readMatrixFromHDFS(file, mb.getNumRows(), mb.getNumColumns(), ConfigurationManager.getBlocksize(), mb.getNonZeros());

			ret.add(new InOut(mb.getInMemorySize(),mbr.getInMemorySize()));
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}


	// private void standardCompressedIOPipelined(int k) {
	// try {
	// // MatrixWriter w = new WriterBinaryBlockParallel(1);
	// // new WriterCompressed(file);
	// MatrixBlock mb = gen.take();
	// WriterCompressed.writeCompressedMatrixToHDFS(mb, file);
	// ret.add(new InOut(mb.getInMemorySize(), getFileSize()));
	// }
	// catch(Exception e) {
	// throw new RuntimeException(e);
	// }
	// }

	private void updateAndApplySchemeFused(CompressionScheme sch, int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock cmb = sch.updateAndEncode(mb, k);
		long out = cmb.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void writeUpdateAndApplySchemeFused(CompressionScheme sch, int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock cmb = sch.updateAndEncode(mb, k);
		Sink o = serialize(cmb);
		ret.add(new InOut(in, o.size()));
	}

	private void diskUpdateAndApplySchemeFused(CompressionScheme sch, int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock cmb = sch.updateAndEncode(mb, k);
		Disk o = serializeD(cmb);
		ret.add(new InOut(in, o.size()));
	}

	private void standardCompressedIOUpdateAndApply(CompressionScheme sch, int k) {
		try {
			// MatrixWriter w = new WriterBinaryBlockParallel(1);
			MatrixBlock mb = gen.take();
			MatrixBlock cmb = sch.updateAndEncode(mb, k);
			WriterCompressed.writeCompressedMatrixToHDFS(cmb, file);
			// ret.add(new InOut(mb.getInMemorySize(), -1));
			ret.add(new InOut(mb.getInMemorySize(), -1));
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	private void setFileSize() {
		try {
			ret.get(ret.size() - 1).out = getFileSize();
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	private void sumTask(int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock r = mb.sum(k);
		long out = r.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private CompressedMatrixBlock getC() throws InterruptedException {
		gen.generate(1);
		MatrixBlock mb = gen.take();
		// System.out.println(mb);
		Pair<MatrixBlock, CompressionStatistics> r = CompressedMatrixBlockFactory.compress(mb);
		// System.out.println(r.getRight());
		return (CompressedMatrixBlock) r.getLeft();
	}

	@Override
	protected String makeResString() {
		throw new RuntimeException("Do not call");
	}

	@Override
	protected String makeResString(double[] times) {
		return makeResString(ret, times);
	}

	public static String makeResString(ArrayList<InOut> ret, double[] times) {
		double totalIn = 0;
		double totalOut = 0;
		double totalTime = 0.0;
		for(int i = 0; i < ret.size(); i++) // set times
			ret.get(i).time = times[i] / 1000; // ms to sec

		ret.sort(Serialize::compare);

		final int l = ret.size();
		final int remove = (int) Math.floor(l * 0.05);

		final int el = l - remove * 2;

		for(int i = remove; i < ret.size() - remove; i++) {
			InOut e = ret.get(i);
			totalIn += e.in;
			totalOut += e.out;
			totalTime += e.time;
		}

		double bytePerMsIn = totalIn / totalTime;
		double bytePerMsOut = totalOut / totalTime;
		// double meanTime = totalTime / el;

		double varIn = 0;
		double varOut = 0;
		// double varTime = 0;

		for(int i = remove; i < ret.size() - remove; i++) {
			InOut e = ret.get(i);
			varIn += Math.pow(e.in / e.time - bytePerMsIn, 2);
			varOut += Math.pow(e.out / e.time - bytePerMsOut, 2);
		}

		double stdIn = Math.sqrt(varIn / el);
		double stdOut = Math.sqrt(varOut / el);

		return String.format("%12.0f+-%12.0f Byte/s, %12.0f+-%12.0f Byte/s", bytePerMsIn, stdIn, bytePerMsOut, stdOut);

	}

	public static int compare(InOut a, InOut b) {
		if(a.time == b.time)
			return 0;
		else if(a.time < b.time)
			return -1;
		else
			return 1;
	}

	public static Sink serialize(MatrixBlock mb) {
		try {
			Sink s = new Sink();
			DataOutputStream fos = new DataOutputStream(s);
			mb.write(fos);
			return s;
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
	}

	private Disk serializeD(MatrixBlock mb) {
		try {
			Disk s = new Disk();
			DataOutputStream fos = new DataOutputStream(s);
			mb.write(fos);
			return s;
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
	}

	private static class Sink extends OutputStream {
		long s = 0L;

		@Override
		public void write(int b) throws IOException {
			s++;
		}

		@Override
		public void write(byte[] b) throws IOException {
			s += b.length;
		}

		public long size() {
			return s;
		}

	}

	private class Disk extends OutputStream {
		final FileOutputStream writer;
		final BufferedOutputStream buf;
		long s = 0L;

		protected Disk() throws FileNotFoundException {
			writer = new FileOutputStream(file);
			buf = new BufferedOutputStream(writer, 4096);
		}

		@Override
		public void write(int b) throws IOException {
			s++;
			buf.write(b);
		}

		@Override
		public void write(byte[] b) throws IOException {
			s += b.length;
			buf.write(b);
		}

		public long size() {
			try {
				buf.close();
				writer.close();
				return s;
			}
			catch(Exception e) {
				return s;
			}
		}
	}

	private void cleanup() {
		File f = new File(file);
		if(f.exists()) {
			if(f.isDirectory())
				deleteDirectory(f);
			else
				f.delete();
		}
		File fd = new File(file + ".dict");
		if(fd.exists()) {
			if(fd.isDirectory())
				deleteDirectory(f);
			else
				fd.delete();
		}

	}

	private boolean deleteDirectory(File directoryToBeDeleted) {
		File[] allContents = directoryToBeDeleted.listFiles();
		if(allContents != null)
			for(File file : allContents) 
				deleteDirectory(file);
		return directoryToBeDeleted.delete();
	}

	private long getFileSize() throws IOException {
		if(new File(file + ".dict").exists())
			return getFileSize(new File(file)) + getFileSize(new File(file + ".dict"));
		else
			return getFileSize(new File(file));
	}

	private long getFileSize(File f) throws IOException {
		if(f.isDirectory()) {
			File[] allContents = f.listFiles();
			long s = 0;
			for(File a : allContents) {
				s += getFileSize(a);
			}
			return s;
		}
		return Files.size(f.toPath());
	}

	@Override
	public String toString() {
		return super.toString() + " threads: " + k;
	}

	protected static class InOut {
		protected long in;
		protected long out;
		protected double time;

		protected InOut(long in, long out) {
			this.in = in;
			this.out = out;
		}

	}

}
