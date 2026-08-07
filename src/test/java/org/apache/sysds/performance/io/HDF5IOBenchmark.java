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

package org.apache.sysds.performance.io;

import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.io.FileFormatPropertiesHDF5;
import org.apache.sysds.runtime.io.ReaderHDF5;
import org.apache.sysds.runtime.io.ReaderHDF5Parallel;
import org.apache.sysds.runtime.io.WriterHDF5;
import org.apache.sysds.runtime.io.WriterHDF5Parallel;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class HDF5IOBenchmark {
	private static final String PROFILE = "sysds.test.hdf5.profile";
	private static final String KEEP = "sysds.test.hdf5.keep.files";
	private static final String LABEL = "sysds.test.hdf5.benchmark.label";
	private static final String DATASET = "DATASET_1";

	private static final int ROWS = Integer.getInteger("sysds.test.hdf5.rows", 250_000);
	private static final int COLS = Integer.getInteger("sysds.test.hdf5.cols", 1_000);
	private static final int BLOCK = Integer.getInteger("sysds.test.hdf5.block.size", 1024);
	private static final int WARMUP = Integer.getInteger("sysds.test.hdf5.warmup.reps", 2);
	private static final int REPS = Integer.getInteger("sysds.test.hdf5.measure.reps", 3);
	private static final int SPARSE_NNZ_PER_ROW = Integer.getInteger("sysds.test.hdf5.sparse.nnz.per.row", 1);

	public static void main(String[] args) throws Exception {
		checkProperties();

		String profile = System.getProperty(PROFILE, "sparse").trim().toLowerCase(Locale.ROOT);
		switch(profile) {
			case "dense":
				benchmarkDense();
				break;
			case "sparse":
				benchmarkSparse();
				break;
			case "all":
				benchmarkDense();
				benchmarkSparse();
				break;
			default:
				throw new IllegalArgumentException(
					"Unsupported HDF5 benchmark profile: " + profile + ". Use dense, sparse, or all.");
		}
	}

	private static void benchmarkDense() throws Exception {
		long nnz = Math.multiplyExact((long) ROWS, (long) COLS);
		MatrixBlock mb = denseMatrix();
		run(new Profile("dense_double", nnz, false), mb);
	}

	private static void benchmarkSparse() throws Exception {
		long nnz = Math.multiplyExact((long) ROWS, (long) SPARSE_NNZ_PER_ROW);
		MatrixBlock mb = sparseMatrix(nnz);
		run(new Profile("sparse_double", nnz, true), mb);
	}

	private static void run(Profile p, MatrixBlock input) throws Exception {
		long cells = Math.multiplyExact((long) ROWS, (long) COLS);
		long logicalDenseBytes = Math.multiplyExact(cells, (long) Double.BYTES);

		Path target = Paths.get("target").toAbsolutePath().normalize();
		Path work = target.resolve("hdf5-bench-" + p.name + "-" + System.currentTimeMillis());
		Files.createDirectories(work);

		Path csv = target.resolve("hdf5-bench-" + p.name + ".csv");
		Path json = target.resolve("hdf5-bench-" + p.name + ".json");

		FileFormatPropertiesHDF5 props = new FileFormatPropertiesHDF5(DATASET);
		WriterHDF5 seqWriter = new WriterHDF5(props);
		WriterHDF5Parallel parWriter = new WriterHDF5Parallel(props);
		ReaderHDF5 seqReader = new ReaderHDF5(props);
		ReaderHDF5Parallel parReader = new ReaderHDF5Parallel(props);

		List<Result> results = new ArrayList<>();
		int total = WARMUP + REPS;

		try {
			for(int rep = 0; rep < total; rep++) {
				boolean warmup = rep < WARMUP;
				int outRep = warmup ? rep : rep - WARMUP;

				Path seqPath = work.resolve(p.name + "_seq_" + rep + ".h5").toAbsolutePath().normalize();
				Path parPath = work.resolve(p.name + "_par_" + rep + ".h5").toAbsolutePath().normalize();

				String seqFile = seqPath.toUri().toString();
				String parFile = parPath.toUri().toString();

				Result seqWrite = measure(p, "seq", "", "hdf5_write", warmup, outRep, seqPath, logicalDenseBytes,
					() -> seqWriter.writeMatrixToHDFS(input, seqFile, ROWS, COLS, BLOCK, p.nnz, false));
				results.add(seqWrite);

				if(seqWrite.ok()) {
					results.add(measure(p, "seq", "seq", "hdf5_read", warmup, outRep, seqPath, logicalDenseBytes,
						() -> validate(p, seqReader.readMatrixFromHDFS(seqFile, ROWS, COLS, BLOCK, p.nnz))));
					results.add(measure(p, "seq", "par", "hdf5_read", warmup, outRep, seqPath, logicalDenseBytes,
						() -> validate(p, parReader.readMatrixFromHDFS(seqFile, ROWS, COLS, BLOCK, p.nnz))));
				}
				else {
					results.add(skip(p, "seq", "seq", "hdf5_read", warmup, outRep, seqPath, logicalDenseBytes));
					results.add(skip(p, "seq", "par", "hdf5_read", warmup, outRep, seqPath, logicalDenseBytes));
				}

				Result parWrite = measure(p, "par", "", "hdf5_write", warmup, outRep, parPath, logicalDenseBytes,
					() -> parWriter.writeMatrixToHDFS(input, parFile, ROWS, COLS, BLOCK, p.nnz, false));
				results.add(parWrite);

				if(parWrite.ok()) {
					results.add(measure(p, "par", "seq", "hdf5_read", warmup, outRep, parPath, logicalDenseBytes,
						() -> validate(p, seqReader.readMatrixFromHDFS(parFile, ROWS, COLS, BLOCK, p.nnz))));
					results.add(measure(p, "par", "par", "hdf5_read", warmup, outRep, parPath, logicalDenseBytes,
						() -> validate(p, parReader.readMatrixFromHDFS(parFile, ROWS, COLS, BLOCK, p.nnz))));
				}
				else {
					results.add(skip(p, "par", "seq", "hdf5_read", warmup, outRep, parPath, logicalDenseBytes));
					results.add(skip(p, "par", "par", "hdf5_read", warmup, outRep, parPath, logicalDenseBytes));
				}
			}
		}
		finally {
			writeCsv(results, csv);
			writeJson(results, json);

			System.out.println("HDF5 benchmark CSV:  " + csv);
			System.out.println("HDF5 benchmark JSON: " + json);
			System.out.println("HDF5 benchmark work: " + work);

			if(!Boolean.parseBoolean(System.getProperty(KEEP, "false")))
				deleteQuietly(work);
		}
	}

	private static Result measure(Profile p, String writer, String reader, String op, boolean warmup, int rep,
		Path path, long logicalDenseBytes, CheckedRunnable action) throws Exception {
		gc();

		long heap0 = usedHeap();
		Gc gc0 = Gc.now();
		long t0 = System.nanoTime();

		Result res = baseResult(p, writer, reader, op, warmup, rep, path, logicalDenseBytes);
		try {
			action.run();
			res.status = "PASS";
		}
		catch(Exception | AssertionError ex) {
			res.status = "FAIL";
			res.errorClass = ex.getClass().getName();
			res.errorMessage = ex.getMessage();
		}

		long t1 = System.nanoTime();
		Gc gc1 = Gc.now();
		long heap1 = usedHeap();

		res.wallMs = (t1 - t0) / 1_000_000.0;
		res.heapBefore = heap0;
		res.heapAfter = heap1;
		res.heapDelta = heap1 - heap0;
		res.gcCount = gc1.count - gc0.count;
		res.gcMs = gc1.ms - gc0.ms;
		res.fileSize = Files.exists(path) ? fileSize(path) : 0;
		res.numFiles = Files.exists(path) ? numFiles(path) : 0;
		return res;
	}

	private static Result skip(Profile p, String writer, String reader, String op, boolean warmup, int rep, Path path,
		long logicalDenseBytes) throws IOException {
		Result res = baseResult(p, writer, reader, op, warmup, rep, path, logicalDenseBytes);
		res.status = "SKIP";
		res.errorMessage = "writer failed";
		res.fileSize = Files.exists(path) ? fileSize(path) : 0;
		res.numFiles = Files.exists(path) ? numFiles(path) : 0;
		return res;
	}

	private static Result baseResult(Profile p, String writer, String reader, String op, boolean warmup, int rep,
		Path path, long logicalDenseBytes) {
		Result r = new Result();
		r.label = System.getProperty(LABEL, "base");
		r.sparseLayout = System.getProperty("sysds.hdf5.write.sparse.layout", "dense");
		r.profile = p.name;
		r.writer = writer;
		r.reader = reader;
		r.operation = op;
		r.status = "RUNNING";
		r.rows = ROWS;
		r.cols = COLS;
		r.cells = Math.multiplyExact((long) ROWS, (long) COLS);
		r.nnz = p.nnz;
		r.sparsity = r.cells == 0 ? 0 : (double) p.nnz / r.cells;
		r.rep = rep;
		r.warmup = warmup;
		r.logicalDenseBytes = logicalDenseBytes;
		r.readParallelism = OptimizerUtils.getParallelBinaryReadParallelism();
		r.writeParallelism = OptimizerUtils.getParallelTextWriteParallelism();
		r.path = path.toString();
		return r;
	}

	private static MatrixBlock denseMatrix() {
		MatrixBlock mb = new MatrixBlock(ROWS, COLS, false);
		mb.allocateDenseBlockUnsafe(ROWS, COLS);

		DenseBlock db = mb.getDenseBlock();
		for(int i = 0; i < ROWS; i++)
			for(int j = 0; j < COLS; j++)
				db.set(i, j, denseValue(i, j));

		mb.setNonZeros(Math.multiplyExact((long) ROWS, (long) COLS));
		mb.examSparsity();
		return mb;
	}

	private static MatrixBlock sparseMatrix(long nnz) {
		MatrixBlock mb = new MatrixBlock(ROWS, COLS, true, nnz);
		mb.allocateSparseRowsBlock();

		SparseBlock sb = mb.getSparseBlock();
		for(int i = 0; i < ROWS; i++) {
			sb.allocate(i, SPARSE_NNZ_PER_ROW);
			for(int j = 0; j < SPARSE_NNZ_PER_ROW; j++)
				sb.append(i, j, sparseValue(i, j));
		}

		mb.setNonZeros(nnz);
		mb.examSparsity();
		check(mb.isInSparseFormat(), "Sparse input converted to dense.");
		return mb;
	}

	private static void validate(Profile p, MatrixBlock mb) {
		checkEquals(ROWS, mb.getNumRows(), "Unexpected number of rows.");
		checkEquals(COLS, mb.getNumColumns(), "Unexpected number of columns.");
		checkEquals(p.nnz, mb.getNonZeros(), "Unexpected number of nonzeros.");

		if(p.sparse) {
			int lastNnzCol = SPARSE_NNZ_PER_ROW - 1;
			check(mb.isInSparseFormat(), "Expected sparse output.");
			checkEquals(sparseValue(0, 0), value(mb, 0, 0), "Unexpected sparse value at first row.");
			checkEquals(sparseValue(ROWS / 2, 0), value(mb, ROWS / 2, 0), "Unexpected sparse value at middle row.");
			checkEquals(sparseValue(ROWS - 1, 0), value(mb, ROWS - 1, 0), "Unexpected sparse value at last row.");
			checkEquals(sparseValue(ROWS / 2, lastNnzCol), value(mb, ROWS / 2, lastNnzCol),
				"Unexpected sparse value at last nonzero column.");

			if(SPARSE_NNZ_PER_ROW < COLS)
				checkEquals(0, value(mb, ROWS - 1, SPARSE_NNZ_PER_ROW), "Expected zero after sparse nonzero range.");
		}
		else {
			check(!mb.isInSparseFormat(), "Expected dense output.");
			checkEquals(denseValue(0, 0), value(mb, 0, 0), "Unexpected dense value at first cell.");
			checkEquals(denseValue(ROWS / 2, COLS / 2), value(mb, ROWS / 2, COLS / 2),
				"Unexpected dense value at middle cell.");
			checkEquals(denseValue(ROWS - 1, COLS - 1), value(mb, ROWS - 1, COLS - 1),
				"Unexpected dense value at last cell.");
		}
	}

	private static double denseValue(int row, int col) {
		return 1.0 + row * 1000.0 + col;
	}

	private static double sparseValue(int row, int col) {
		return col < SPARSE_NNZ_PER_ROW ? denseValue(row, col) : 0.0;
	}

	private static double value(MatrixBlock mb, int row, int col) {
		if(!mb.isInSparseFormat())
			return mb.getDenseBlock().get(row, col);

		SparseBlock sb = mb.getSparseBlock();
		if(sb == null || sb.isEmpty(row))
			return 0;

		int pos = sb.pos(row);
		int end = pos + sb.size(row);
		int[] ix = sb.indexes(row);
		double[] vals = sb.values(row);

		for(int p = pos; p < end; p++)
			if(ix[p] == col)
				return vals[p];
		return 0;
	}

	private static void checkProperties() {
		if(ROWS <= 0 || COLS <= 0 || BLOCK <= 0)
			throw new IllegalArgumentException("rows, cols, and block size must be positive.");
		if(WARMUP < 0 || REPS <= 0)
			throw new IllegalArgumentException("warmup must be >= 0 and reps must be > 0.");
		if(SPARSE_NNZ_PER_ROW <= 0 || SPARSE_NNZ_PER_ROW > COLS)
			throw new IllegalArgumentException("sparse.nnz.per.row must be in [1, cols].");
	}

	private static void check(boolean condition, String message) {
		if(!condition)
			throw new IllegalStateException(message);
	}

	private static void checkEquals(long expected, long actual, String message) {
		if(expected != actual)
			throw new IllegalStateException(message + " Expected=" + expected + ", actual=" + actual + ".");
	}

	private static void checkEquals(double expected, double actual, String message) {
		if(Double.compare(expected, actual) != 0)
			throw new IllegalStateException(message + " Expected=" + expected + ", actual=" + actual + ".");
	}

	private static long usedHeap() {
		Runtime rt = Runtime.getRuntime();
		return rt.totalMemory() - rt.freeMemory();
	}

	private static void gc() {
		System.gc();
		try {
			Thread.sleep(100);
		}
		catch(InterruptedException ex) {
			Thread.currentThread().interrupt();
		}
	}

	private static long fileSize(Path p) throws IOException {
		if(Files.isRegularFile(p))
			return Files.size(p);
		if(!Files.isDirectory(p))
			return 0;

		final long[] ret = new long[] {0};
		try(Stream<Path> s = Files.walk(p)) {
			s.filter(Files::isRegularFile).forEach(x -> {
				try {
					ret[0] += Files.size(x);
				}
				catch(IOException ex) {
					throw new RuntimeException(ex);
				}
			});
		}
		return ret[0];
	}

	private static int numFiles(Path p) throws IOException {
		if(Files.isRegularFile(p))
			return 1;
		if(!Files.isDirectory(p))
			return 0;

		final int[] ret = new int[] {0};
		try(Stream<Path> s = Files.walk(p)) {
			s.filter(Files::isRegularFile).forEach(x -> ret[0]++);
		}
		return ret[0];
	}

	private static void deleteQuietly(Path p) {
		try {
			if(!Files.exists(p))
				return;
			try(Stream<Path> s = Files.walk(p)) {
				s.sorted(Comparator.reverseOrder()).forEach(x -> {
					try {
						Files.deleteIfExists(x);
					}
					catch(IOException ex) {
						throw new RuntimeException(ex);
					}
				});
			}
		}
		catch(Exception ex) {
			System.err.println("Could not delete benchmark directory: " + p);
		}
	}

	private static void writeCsv(List<Result> results, Path out) throws IOException {
		StringBuilder sb = new StringBuilder();
		sb.append("label,sparse_layout,profile,writer,reader,operation,status,rep,warmup,rows,cols,cells,nnz,sparsity,")
			.append("wall_ms,file_size,num_files,logical_dense_bytes,heap_before,heap_after,heap_delta,")
			.append("gc_count,gc_ms,read_parallelism,write_parallelism,path,error_class,error_message\n");

		for(Result r : results)
			sb.append(csv(r.label)).append(',').append(csv(r.sparseLayout)).append(',').append(csv(r.profile))
				.append(',').append(csv(r.writer)).append(',').append(csv(r.reader)).append(',')
				.append(csv(r.operation)).append(',').append(csv(r.status)).append(',').append(r.rep).append(',')
				.append(r.warmup).append(',').append(r.rows).append(',').append(r.cols).append(',').append(r.cells)
				.append(',').append(r.nnz).append(',').append(String.format(Locale.US, "%.8f", r.sparsity)).append(',')
				.append(String.format(Locale.US, "%.3f", r.wallMs)).append(',').append(r.fileSize).append(',')
				.append(r.numFiles).append(',').append(r.logicalDenseBytes).append(',').append(r.heapBefore).append(',')
				.append(r.heapAfter).append(',').append(r.heapDelta).append(',').append(r.gcCount).append(',')
				.append(r.gcMs).append(',').append(r.readParallelism).append(',').append(r.writeParallelism).append(',')
				.append(csv(r.path)).append(',').append(csv(r.errorClass)).append(',').append(csv(r.errorMessage))
				.append('\n');

		Files.write(out, sb.toString().getBytes(StandardCharsets.UTF_8));
	}

	private static void writeJson(List<Result> results, Path out) throws IOException {
		StringBuilder sb = new StringBuilder("[\n");
		for(int i = 0; i < results.size(); i++) {
			if(i > 0)
				sb.append(",\n");
			sb.append(results.get(i).json());
		}
		sb.append("\n]\n");
		Files.write(out, sb.toString().getBytes(StandardCharsets.UTF_8));
	}

	private static String csv(String s) {
		return s == null ? "" : "\"" + s.replace("\"", "\"\"") + "\"";
	}

	private static String json(String s) {
		return s == null ? "null" : "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
	}

	private interface CheckedRunnable {
		void run() throws Exception;
	}

	private static class Profile {
		final String name;
		final long nnz;
		final boolean sparse;

		Profile(String name, long nnz, boolean sparse) {
			this.name = name;
			this.nnz = nnz;
			this.sparse = sparse;
		}
	}

	private static class Gc {
		long count;
		long ms;

		static Gc now() {
			Gc g = new Gc();
			for(GarbageCollectorMXBean b : ManagementFactory.getGarbageCollectorMXBeans()) {
				long c = b.getCollectionCount();
				long t = b.getCollectionTime();
				if(c > 0)
					g.count += c;
				if(t > 0)
					g.ms += t;
			}
			return g;
		}
	}

	private static class Result {
		String label, sparseLayout, profile, writer, reader, operation, status, path, errorClass, errorMessage;
		int rows, cols, rep, numFiles, readParallelism, writeParallelism;
		long cells, nnz, fileSize, logicalDenseBytes, heapBefore, heapAfter, heapDelta, gcCount, gcMs;
		boolean warmup;
		double sparsity, wallMs;

		boolean ok() {
			return "PASS".equals(status);
		}

		String json() {
			return "{" + "\"label\":" + HDF5IOBenchmark.json(label) + ",\"sparse_layout\":"
				+ HDF5IOBenchmark.json(sparseLayout) + ",\"profile\":" + HDF5IOBenchmark.json(profile) + ",\"writer\":"
				+ HDF5IOBenchmark.json(writer) + ",\"reader\":" + HDF5IOBenchmark.json(reader) + ",\"operation\":"
				+ HDF5IOBenchmark.json(operation) + ",\"status\":" + HDF5IOBenchmark.json(status) + ",\"rep\":" + rep
				+ ",\"warmup\":" + warmup + ",\"rows\":" + rows + ",\"cols\":" + cols + ",\"cells\":" + cells
				+ ",\"nnz\":" + nnz + ",\"sparsity\":" + String.format(Locale.US, "%.8f", sparsity) + ",\"wall_ms\":"
				+ String.format(Locale.US, "%.3f", wallMs) + ",\"file_size\":" + fileSize + ",\"num_files\":" + numFiles
				+ ",\"logical_dense_bytes\":" + logicalDenseBytes + ",\"heap_before\":" + heapBefore
				+ ",\"heap_after\":" + heapAfter + ",\"heap_delta\":" + heapDelta + ",\"gc_count\":" + gcCount
				+ ",\"gc_ms\":" + gcMs + ",\"read_parallelism\":" + readParallelism + ",\"write_parallelism\":"
				+ writeParallelism + ",\"path\":" + HDF5IOBenchmark.json(path) + ",\"error_class\":"
				+ HDF5IOBenchmark.json(errorClass) + ",\"error_message\":" + HDF5IOBenchmark.json(errorMessage) + "}";
		}
	}
}
