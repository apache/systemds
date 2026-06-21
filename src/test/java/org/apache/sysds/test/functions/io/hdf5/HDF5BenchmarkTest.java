/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.io.hdf5;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Assume;
import org.junit.Test;

/**
 * Manual HDF5 I/O benchmark.
 *
 * This test is disabled by default and only runs with
 * -Dsysds.test.hdf5.benchmark=true. It writes CSV/JSON results under target/ and is intended for local
 * performance diagnostics, not CI performance checking.
 */
public class HDF5BenchmarkTest {
    private static final String ENABLE_PROPERTY = "sysds.test.hdf5.benchmark";
    private static final String KEEP_FILES_PROPERTY = "sysds.test.hdf5.keep.files";

    private static final String DATASET_NAME = "DATASET_1";

    private static final int ROWS = Integer.getInteger("sysds.test.hdf5.rows", 10_000);
    private static final int COLS = Integer.getInteger("sysds.test.hdf5.cols", 100);
    private static final int BLOCK_SIZE = Integer.getInteger("sysds.test.hdf5.block.size", 1024);

    private static final int WARMUP_REPS = Integer.getInteger("sysds.test.hdf5.warmup.reps", 1);
    private static final int MEASURE_REPS = Integer.getInteger("sysds.test.hdf5.measure.reps", 3);
    private static final int RAW_BUFFER_SIZE = Integer.getInteger("sysds.test.hdf5.raw.buffer.bytes", 8 * 1024 * 1024);

    private static final int SPARSE_NNZ_PER_ROW = Integer.getInteger("sysds.test.hdf5.sparse.nnz.per.row",
            Math.max(1, COLS / 100));

    private static final byte RAW_BYTE_VALUE = (byte) 7;

    @Test
    public void benchmarkHDF5ReadWrite() throws Exception {
        Assume.assumeTrue("HDF5 benchmark disabled. Enable with -D" + ENABLE_PROPERTY + "=true", Boolean.parseBoolean(System.getProperty(ENABLE_PROPERTY, "false")));

        validateBenchmarkProperties();

        final long cells = Math.multiplyExact((long) ROWS, (long) COLS);
        final long denseDoubleReferenceSizeBytes = Math.multiplyExact(cells, (long) Double.BYTES);

        Path targetDir = Paths.get("target").toAbsolutePath().normalize();
        Files.createDirectories(targetDir);

        Path workDir = targetDir.resolve("hdf5-benchmark-work-" + System.currentTimeMillis());
        Files.createDirectories(workDir);

        Path csvOut = targetDir.resolve("hdf5-benchmark-results.csv");
        Path jsonOut = targetDir.resolve("hdf5-benchmark-results.json");

        boolean success = false;
        try {
            FileFormatPropertiesHDF5 props = new FileFormatPropertiesHDF5(DATASET_NAME);

            WriterHDF5 writer = new WriterHDF5(props);
            ReaderHDF5 sequentialReader = new ReaderHDF5(props);
            ReaderHDF5Parallel parallelReader = new ReaderHDF5Parallel(props);

            List<DataProfile> profiles = new ArrayList<>();
            profiles.add(createDenseProfile(cells));
            profiles.add(createSparseLikeProfile());

            List<Result> results = new ArrayList<>();

            int totalReps = WARMUP_REPS + MEASURE_REPS;
            for(DataProfile profile : profiles) {
                for(int rep = 0; rep < totalReps; rep++) {
                    boolean warmup = rep < WARMUP_REPS;
                    int logicalRep = warmup ? rep : rep - WARMUP_REPS;

                    Path rawPath = workDir.resolve(profile.name + "_raw_rep_" + rep + ".bin").toAbsolutePath().normalize();
                    Path hdf5Path = workDir.resolve(profile.name + "_hdf5_rep_" + rep + ".h5").toAbsolutePath().normalize();

                    String hdf5Filename = hdf5Path.toUri().toString();

                    Result rawWriteResult = measure(
                            profile.name,
                            "raw",
                            "raw_write",
                            warmup,
                            logicalRep,
                            rawPath,
                            denseDoubleReferenceSizeBytes,
                            profile.logicalNnz,
                            "Chunked raw byte write; dense-Double-size I/O control; no HDF5 encoding; no MatrixBlock materialization",
                            new ThrowingRunnable() {
                                @Override
                                public void run() throws Exception {
                                    writeRawBytes(rawPath, denseDoubleReferenceSizeBytes, RAW_BUFFER_SIZE);
                                }
                            }
                    );
                    results.add(rawWriteResult);

                    assertTrue("Raw file was not created: " + rawPath, Files.exists(rawPath));
                    assertEquals("Raw file size mismatch.", denseDoubleReferenceSizeBytes, Files.size(rawPath));

                    Result rawReadResult = measure(
                            profile.name,
                            "raw",
                            "raw_read",
                            warmup,
                            logicalRep,
                            rawPath,
                            denseDoubleReferenceSizeBytes,
                            profile.logicalNnz,
                            "Chunked raw byte read; dense-Double-size I/O control; no HDF5 decoding; no MatrixBlock materialization",
                            new ThrowingRunnable() {
                                @Override
                                public void run() throws Exception {
                                    readRawBytes(rawPath, denseDoubleReferenceSizeBytes, RAW_BUFFER_SIZE);
                                }
                            }
                    );
                    results.add(rawReadResult);

                    Result hdf5WriteResult = measure(
                            profile.name,
                            "seq",
                            "hdf5_write",
                            warmup,
                            logicalRep,
                            hdf5Path,
                            denseDoubleReferenceSizeBytes,
                            profile.logicalNnz,
                            profile.writePathNote,
                            new ThrowingRunnable() {
                                @Override
                                public void run() throws Exception {
                                    writer.writeMatrixToHDFS(profile.matrix, hdf5Filename, ROWS, COLS, BLOCK_SIZE, profile.logicalNnz, false);
                                }
                            }
                    );
                    results.add(hdf5WriteResult);

                    assertTrue("HDF5 file was not created: " + hdf5Path, Files.exists(hdf5Path));
                    assertTrue("HDF5 file size must be positive", totalFileSizeBytes(hdf5Path) > 0);

                    Result hdf5SeqReadResult = measure(
                            profile.name,
                            "seq",
                            "hdf5_read",
                            warmup,
                            logicalRep,
                            hdf5Path,
                            denseDoubleReferenceSizeBytes,
                            profile.logicalNnz,
                            profile.sequentialReadPathNote,
                            new ThrowingRunnable() {
                                @Override
                                public void run() throws Exception {
                                    MatrixBlock out = sequentialReader.readMatrixFromHDFS(hdf5Filename, ROWS, COLS, BLOCK_SIZE, profile.logicalNnz);
                                    validateProfileMatrix(profile, out);
                                }
                            }
                    );
                    results.add(hdf5SeqReadResult);

                    Result hdf5ParReadResult = measure(
                            profile.name,
                            "par",
                            "hdf5_read",
                            warmup,
                            logicalRep,
                            hdf5Path,
                            denseDoubleReferenceSizeBytes,
                            profile.logicalNnz,
                            profile.parallelReadPathNote,
                            new ThrowingRunnable() {
                                @Override
                                public void run() throws Exception {
                                    MatrixBlock out = parallelReader.readMatrixFromHDFS(hdf5Filename, ROWS, COLS, BLOCK_SIZE, profile.logicalNnz);
                                    validateProfileMatrix(profile, out);
                                }
                            }
                    );
                    results.add(hdf5ParReadResult);
                }
            }

            writeCsv(results, csvOut);
            writeJson(results, jsonOut);

            System.out.println("HDF5 benchmark CSV:  " + csvOut);
            System.out.println("HDF5 benchmark JSON: " + jsonOut);
            System.out.println("HDF5 benchmark work: " + workDir);
            success = true;
        }
        finally {
            if(success && !Boolean.parseBoolean(System.getProperty(KEEP_FILES_PROPERTY, "false")))
                deleteRecursivelySiltently(workDir);
        }
    }

    private static void validateBenchmarkProperties() {
        if(ROWS <= 0)
            throw new IllegalArgumentException("Invalid sysds.test.hdf5.rows=" + ROWS + ". Must be > 0.");
        if(COLS <= 0)
            throw new IllegalArgumentException("Invalid sysds.test.hdf5.cols=" + COLS + ". Must be > 0.");
        if(BLOCK_SIZE <= 0)
            throw new IllegalArgumentException("Invalid sysds.test.hdf5.block.size=" + BLOCK_SIZE + ". Must be > 0.");
        if(WARMUP_REPS < 0)
            throw new IllegalArgumentException("Invalid sysds.test.hdf5.warmup.reps=" + WARMUP_REPS + ". Must be >= 0.");
        if(MEASURE_REPS <= 0)
            throw new IllegalArgumentException("Invalid sysds.test.hdf5.measure.reps=" + MEASURE_REPS + ". Must be > 0.");
        if(RAW_BUFFER_SIZE <= 0)
            throw new IllegalArgumentException(
                    "Invalid sysds.test.hdf5.raw.buffer.bytes=" + RAW_BUFFER_SIZE + ". Must be > 0.");
        if(SPARSE_NNZ_PER_ROW <= 0 || SPARSE_NNZ_PER_ROW > COLS) {
            throw new IllegalArgumentException("Invalid sysds.test.hdf5.sparse.nnz.per.row=" + SPARSE_NNZ_PER_ROW + ". Must be in [1, COLS]. COLS=" + COLS);
        }
    }

    private static DataProfile createDenseProfile(long cells) {
        MatrixBlock matrixBlock = new MatrixBlock(ROWS, COLS, false);
        matrixBlock.allocateDenseBlockUnsafe(ROWS, COLS);

        DenseBlock db = matrixBlock.getDenseBlock();
        for(int i = 0; i < ROWS; i++) {
            for(int j = 0; j < COLS; j++)
                db.set(i, j, expectedDenseValue(i, j));
        }

        matrixBlock.recomputeNonZeros();
        matrixBlock.examSparsity();

        DataProfile p = new DataProfile();
        p.name = "dense_double_only";
        p.matrix = matrixBlock;
        p.logicalNnz = cells;
        p.expectedSparse = false;
        p.writePathNote = "WriterHDF5.writeMatrixToHDFS(MatrixBlock,...); dense MatrixBlock write path";
        p.sequentialReadPathNote = "ReaderHDF5.readMatrixFromHDFS(String,...); dense double MatrixBlock read path";
        p.parallelReadPathNote = "ReaderHDF5Parallel.readMatrixFromHDFS(String,...)";
        return p;
    }

    private static DataProfile createSparseLikeProfile() {
        long nnz = Math.multiplyExact((long) ROWS, (long) SPARSE_NNZ_PER_ROW);

        MatrixBlock matrixBlock = new MatrixBlock(ROWS, COLS, true, nnz);
        matrixBlock.allocateSparseRowsBlock();

        SparseBlock sparseBlock = matrixBlock.getSparseBlock();
        for(int i = 0; i < ROWS; i++) {
            sparseBlock.allocate(i, SPARSE_NNZ_PER_ROW);
            for(int j = 0; j < SPARSE_NNZ_PER_ROW; j++)
                sparseBlock.append(i, j, expectedSparseLikeValue(i, j));
        }

        matrixBlock.setNonZeros(nnz);
        matrixBlock.examSparsity();

        assertTrue("Sparse-like input unexpectedly converted to dense. Reduce sparse.nnz.per.row.", matrixBlock.isInSparseFormat());

        DataProfile p = new DataProfile();
        p.name = "sparse_like_double";
        p.matrix = matrixBlock;
        p.logicalNnz = nnz;
        p.expectedSparse = true;
        p.writePathNote = "WriterHDF5.writeMatrixToHDFS(MatrixBlock,...)";
        p.sequentialReadPathNote = "ReaderHDF5.readMatrixFromHDFS(String,...)";
        p.parallelReadPathNote = "ReaderHDF5Parallel.readMatrixFromHDFS(String,...)";
        return p;
    }

    private static Result measure(
            String dataProfile,
            String impl,
            String operation,
            boolean warmup,
            int rep,
            Path measuredPath,
            long denseDoubleReferenceSizeBytes,
            long logicalNnz,
            String implementationPath,
            ThrowingRunnable runnable
    ) throws Exception {
        long heapBefore = usedHeapBytes();
        GcSnapshot gcBefore = GcSnapshot.capture();

        long t0 = System.nanoTime();
        runnable.run();
        long t1 = System.nanoTime();

        GcSnapshot gcAfter = GcSnapshot.capture();
        long heapAfter = usedHeapBytes();

        long fileSize = Files.exists(measuredPath) ? totalFileSizeBytes(measuredPath) : 0L;
        int numPartFiles = Files.exists(measuredPath) ? countRegularFiles(measuredPath) : 0;

        Result r = new Result();
        r.dataProfile = dataProfile;
        r.impl = impl;
        r.operation = operation;
        r.rows = ROWS;
        r.cols = COLS;
        r.cells = Math.multiplyExact((long) ROWS, (long) COLS);
        r.logicalNnz = logicalNnz;
        r.logicalSparsity = r.cells > 0 ? ((double) logicalNnz) / r.cells : 0.0;
        r.sparseNnzPerRow = dataProfile.equals("sparse_like_double") ? SPARSE_NNZ_PER_ROW : COLS;
        r.rep = rep;
        r.isWarmup = warmup;
        r.wallMs = (t1 - t0) / 1_000_000.0;
        r.fileSizeBytes = fileSize;
        r.numPartFiles = numPartFiles;
        r.avgPartFileSizeBytes = numPartFiles > 0 ? ((double) fileSize) / numPartFiles : 0.0;
        r.denseDoubleReferenceSizeBytes = denseDoubleReferenceSizeBytes;
        r.actualFileToDenseDoubleReferenceRatio = denseDoubleReferenceSizeBytes > 0 ? ((double) fileSize) / denseDoubleReferenceSizeBytes : 0.0;
        r.heapBeforeBytes = heapBefore;
        r.heapAfterBytes = heapAfter;
        r.heapDeltaBytes = heapAfter - heapBefore;
        r.gcCountDelta = gcAfter.count - gcBefore.count;
        r.gcTimeDeltaMs = gcAfter.timeMs - gcBefore.timeMs;
        r.configuredParallelReadParallelism = OptimizerUtils.getParallelBinaryReadParallelism();
        r.configuredParallelWriteParallelism = OptimizerUtils.getParallelTextWriteParallelism();
        r.rawBufferBytesProperty = Integer.toString(RAW_BUFFER_SIZE);
        r.keepFilesProperty = System.getProperty(KEEP_FILES_PROPERTY, "false");
        r.hdf5ReadParallelThreadsProperty = System.getProperty("sysds.hdf5.read.parallel.threads", "");
        r.hdf5ReadParallelMinBytesProperty = System.getProperty("sysds.hdf5.read.parallel.min.bytes", "");
        r.hdf5ReadBlockBytesProperty = System.getProperty("sysds.hdf5.read.block.bytes", "");
        r.hdf5ReadBufferBytesProperty = System.getProperty("sysds.hdf5.read.buffer.bytes", "");
        r.hdf5ReadMapBytesProperty = System.getProperty("sysds.hdf5.read.map.bytes", "");
        r.hdf5ReadMmapProperty = System.getProperty("sysds.hdf5.read.mmap", "");
        r.hdf5ReadTraceProperty = System.getProperty("sysds.hdf5.read.trace", "");
        r.hdf5SkipNnzProperty = System.getProperty("sysds.hdf5.read.skip.nnz", "");
        r.hdf5ForceDenseProperty = System.getProperty("sysds.hdf5.read.force.dense", "");
        r.implementationPath = implementationPath;
        return r;
    }

    private static double expectedDenseValue(int row, int col) {
        return 1.0 + ((double) row * 1000.0) + (double) col;
    }

    private static double expectedSparseLikeValue(int row, int col) {
        if(col < SPARSE_NNZ_PER_ROW)
            return 1.0 + ((double) row * 1000.0) + (double) col;
        return 0.0;
    }

    private static void validateProfileMatrix(DataProfile profile, MatrixBlock matrixBlock) {
        assertEquals(ROWS, matrixBlock.getNumRows());
        assertEquals(COLS, matrixBlock.getNumColumns());
        assertEquals(profile.logicalNnz, matrixBlock.getNonZeros());

        if(profile.expectedSparse) {
            assertTrue("Sparse-like output unexpectedly dense.", matrixBlock.isInSparseFormat());

            assertEquals(expectedSparseLikeValue(0, 0), getMatrixValue(matrixBlock, 0, 0), 0.0);
            assertEquals(expectedSparseLikeValue(ROWS / 2, 0), getMatrixValue(matrixBlock, ROWS / 2, 0), 0.0);
            assertEquals(expectedSparseLikeValue(ROWS - 1, 0), getMatrixValue(matrixBlock, ROWS - 1, 0), 0.0);

            if(SPARSE_NNZ_PER_ROW < COLS) {
                assertEquals(0.0, getMatrixValue(matrixBlock, 0, COLS - 1), 0.0);
                assertEquals(0.0, getMatrixValue(matrixBlock, ROWS / 2, COLS - 1), 0.0);
                assertEquals(0.0, getMatrixValue(matrixBlock, ROWS - 1, COLS - 1), 0.0);
            }
        }
        else {
            assertTrue("Dense Double output unexpectedly sparse.", !matrixBlock.isInSparseFormat());

            assertEquals(expectedDenseValue(0, 0), getMatrixValue(matrixBlock, 0, 0), 0.0);
            assertEquals(expectedDenseValue(ROWS / 2, COLS / 2), getMatrixValue(matrixBlock, ROWS / 2, COLS / 2), 0.0);
            assertEquals(expectedDenseValue(ROWS - 1, COLS - 1), getMatrixValue(matrixBlock, ROWS - 1, COLS - 1), 0.0);
        }
    }

    private static double getMatrixValue(MatrixBlock matrixBlock, int row, int col) {
        if(!matrixBlock.isInSparseFormat())
            return matrixBlock.getDenseBlock().get(row, col);

        SparseBlock sparseBlock = matrixBlock.getSparseBlock();
        if(sparseBlock == null || sparseBlock.isEmpty(row))
            return 0.0;

        int rowStartPosition = sparseBlock.pos(row);
        int rowNonZeroCount = sparseBlock.size(row);
        int[] columnIndexes = sparseBlock.indexes(row);
        double[] nonZeroValues = sparseBlock.values(row);

        for(int k = rowStartPosition; k < rowStartPosition + rowNonZeroCount; k++) {
            if(columnIndexes[k] == col)
                return nonZeroValues[k];
        }

        return 0.0;
    }

    private static void writeRawBytes(Path path, long totalBytes, int bufferSize) throws IOException {
        byte[] buffer = new byte[bufferSize];
        Arrays.fill(buffer, RAW_BYTE_VALUE);

        try(OutputStream outputStream = new BufferedOutputStream(
                Files.newOutputStream(
                        path, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE),
                bufferSize)) {
            long written = 0;
            while(written < totalBytes) {
                int len = (int) Math.min(buffer.length, totalBytes - written);
                outputStream.write(buffer, 0, len);
                written += len;
            }
        }
    }

    private static void readRawBytes(Path path, long expectedBytes, int bufferSize) throws IOException {
        byte[] buffer = new byte[bufferSize];

        try(InputStream inputStream = new BufferedInputStream(Files.newInputStream(path, StandardOpenOption.READ), bufferSize)) {
            long total = 0;
            int len;
            while((len = inputStream.read(buffer)) >= 0)
                total += len;
            assertEquals("Raw byte read length mismatch.", expectedBytes, total);
        }
    }

    private static long usedHeapBytes() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static long totalFileSizeBytes(Path p) throws IOException {
        if(Files.isRegularFile(p))
            return Files.size(p);
        if(!Files.isDirectory(p))
            return 0L;

        final long[] size = new long[] {0L};
        try(Stream<Path> stream = Files.walk(p)) {
            stream.filter(Files::isRegularFile).forEach(x -> {
                try {
                    size[0] += Files.size(x);
                }
                catch(IOException exception) {
                    throw new RuntimeException(exception);
                }
            });
        }
        return size[0];
    }

    private static int countRegularFiles(Path p) throws IOException {
        if(Files.isRegularFile(p))
            return 1;
        if(!Files.isDirectory(p))
            return 0;

        final int[] count = new int[] {0};
        try(Stream<Path> stream = Files.walk(p)) {
            stream.filter(Files::isRegularFile).forEach(x -> count[0]++);
        }
        return count[0];
    }

    private static void deleteRecursively(Path p) throws IOException {
        if(!Files.exists(p))
            return;

        try(Stream<Path> stream = Files.walk(p)) {
            stream.sorted(Comparator.reverseOrder()).forEach(x -> {
                try {
                    Files.deleteIfExists(x);
                }
                catch(IOException exception) {
                    throw new RuntimeException(exception);
                }
            });
        }
    }

    private static void deleteRecursivelySiltently(Path p) {
        final int maxAttempts = 3;

        for(int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                deleteRecursively(p);
                return;
            }
            catch(IOException | RuntimeException exception) {
                if(attempt == maxAttempts) {
                    System.err.println("Could not delete HDF5 benchmark work directory: " + p);
                    System.err.println("This does not invalidate the benchmark results. "
                            + "On Windows, HDF5 or memory-mapped I/O can keep files locked briefly.");
                    System.err.println("Delete the directory manually later, or run with -D" + KEEP_FILES_PROPERTY + "=true.");
                    System.err.println("Cleanup error: " + exception.getMessage());
                    return;
                }

                System.gc();
                try {
                    Thread.sleep(250L);
                }
                catch(InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    private static void writeCsv(List<Result> results, Path out) throws IOException {
        StringBuilder stringBuilder = new StringBuilder();

        stringBuilder.append("data_profile,impl,operation,rows,cols,cells,logical_nnz,logical_sparsity,sparse_nnz_per_row,")
                .append("rep,is_warmup,wall_ms,")
                .append("file_size_bytes,num_part_files,avg_part_file_size_bytes,")
                .append("dense_double_reference_size_bytes,actual_file_to_dense_double_reference_ratio,")
                .append("heap_before_bytes,heap_after_bytes,heap_delta_bytes,")
                .append("gc_count_delta,gc_time_delta_ms,")
                .append("configured_parallel_read_parallelism,configured_parallel_write_parallelism,")
                .append("raw_buffer_bytes_property,keep_files_property,")
                .append("hdf5_read_parallel_threads_property,hdf5_read_parallel_min_bytes_property,")
                .append("hdf5_read_block_bytes_property,hdf5_read_buffer_bytes_property,hdf5_read_map_bytes_property,")
                .append("hdf5_read_mmap_property,hdf5_read_trace_property,hdf5_skip_nnz_property,hdf5_force_dense_property,")
                .append("implementation_path\n");

        for(Result r : results) {
            stringBuilder.append(csv(r.dataProfile)).append(',')
                    .append(csv(r.impl)).append(',')
                    .append(csv(r.operation)).append(',')
                    .append(r.rows).append(',')
                    .append(r.cols).append(',')
                    .append(r.cells).append(',')
                    .append(r.logicalNnz).append(',')
                    .append(String.format(Locale.US, "%.8f", r.logicalSparsity)).append(',')
                    .append(r.sparseNnzPerRow).append(',')
                    .append(r.rep).append(',')
                    .append(r.isWarmup).append(',')
                    .append(String.format(Locale.US, "%.3f", r.wallMs)).append(',')
                    .append(r.fileSizeBytes).append(',')
                    .append(r.numPartFiles).append(',')
                    .append(String.format(Locale.US, "%.3f", r.avgPartFileSizeBytes)).append(',')
                    .append(r.denseDoubleReferenceSizeBytes).append(',')
                    .append(String.format(Locale.US, "%.6f", r.actualFileToDenseDoubleReferenceRatio)).append(',')
                    .append(r.heapBeforeBytes).append(',')
                    .append(r.heapAfterBytes).append(',')
                    .append(r.heapDeltaBytes).append(',')
                    .append(r.gcCountDelta).append(',')
                    .append(r.gcTimeDeltaMs).append(',')
                    .append(r.configuredParallelReadParallelism).append(',')
                    .append(r.configuredParallelWriteParallelism).append(',')
                    .append(csv(r.rawBufferBytesProperty)).append(',')
                    .append(csv(r.keepFilesProperty)).append(',')
                    .append(csv(r.hdf5ReadParallelThreadsProperty)).append(',')
                    .append(csv(r.hdf5ReadParallelMinBytesProperty)).append(',')
                    .append(csv(r.hdf5ReadBlockBytesProperty)).append(',')
                    .append(csv(r.hdf5ReadBufferBytesProperty)).append(',')
                    .append(csv(r.hdf5ReadMapBytesProperty)).append(',')
                    .append(csv(r.hdf5ReadMmapProperty)).append(',')
                    .append(csv(r.hdf5ReadTraceProperty)).append(',')
                    .append(csv(r.hdf5SkipNnzProperty)).append(',')
                    .append(csv(r.hdf5ForceDenseProperty)).append(',')
                    .append(csv(r.implementationPath)).append('\n');
        }

        Files.write(out, stringBuilder.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void writeJson(List<Result> results, Path out) throws IOException {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("[\n");
        for(int i = 0; i < results.size(); i++) {
            if(i > 0)
                stringBuilder.append(",\n");
            stringBuilder.append(results.get(i).toJson());
        }
        stringBuilder.append("\n]\n");

        Files.write(out, stringBuilder.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static String csv(String s) {
        if(s == null)
            return "";
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    private static String json(String s) {
        if(s == null)
            return "null";
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private interface ThrowingRunnable {
        void run() throws Exception;
    }

    private static class DataProfile {
        String name;
        MatrixBlock matrix;
        long logicalNnz;
        boolean expectedSparse;
        String writePathNote;
        String sequentialReadPathNote;
        String parallelReadPathNote;
    }

    private static class GcSnapshot {
        long count;
        long timeMs;

        static GcSnapshot capture() {
            GcSnapshot snapshot = new GcSnapshot();
            for(GarbageCollectorMXBean bean : ManagementFactory.getGarbageCollectorMXBeans()) {
                long c = bean.getCollectionCount();
                long t = bean.getCollectionTime();

                if(c > 0)
                    snapshot.count += c;
                if(t > 0)
                    snapshot.timeMs += t;
            }
            return snapshot;
        }
    }

    private static class Result {
        String dataProfile;
        String impl;
        String operation;
        int rows;
        int cols;
        long cells;
        long logicalNnz;
        double logicalSparsity;
        int sparseNnzPerRow;
        int rep;
        boolean isWarmup;
        double wallMs;

        long fileSizeBytes;
        int numPartFiles;
        double avgPartFileSizeBytes;
        long denseDoubleReferenceSizeBytes;
        double actualFileToDenseDoubleReferenceRatio;

        long heapBeforeBytes;
        long heapAfterBytes;
        long heapDeltaBytes;
        long gcCountDelta;
        long gcTimeDeltaMs;

        int configuredParallelReadParallelism;
        int configuredParallelWriteParallelism;

        String rawBufferBytesProperty;
        String keepFilesProperty;
        String hdf5ReadParallelThreadsProperty;
        String hdf5ReadParallelMinBytesProperty;
        String hdf5ReadBlockBytesProperty;
        String hdf5ReadBufferBytesProperty;
        String hdf5ReadMapBytesProperty;
        String hdf5ReadMmapProperty;
        String hdf5ReadTraceProperty;
        String hdf5SkipNnzProperty;
        String hdf5ForceDenseProperty;

        String implementationPath;

        String toJson() {
            return "{"
                    + "\"data_profile\":" + json(dataProfile)
                    + ",\"impl\":" + json(impl)
                    + ",\"operation\":" + json(operation)
                    + ",\"rows\":" + rows
                    + ",\"cols\":" + cols
                    + ",\"cells\":" + cells
                    + ",\"logical_nnz\":" + logicalNnz
                    + ",\"logical_sparsity\":" + String.format(Locale.US, "%.8f", logicalSparsity)
                    + ",\"sparse_nnz_per_row\":" + sparseNnzPerRow
                    + ",\"rep\":" + rep
                    + ",\"is_warmup\":" + isWarmup
                    + ",\"wall_ms\":" + String.format(Locale.US, "%.3f", wallMs)
                    + ",\"file_size_bytes\":" + fileSizeBytes
                    + ",\"num_part_files\":" + numPartFiles
                    + ",\"avg_part_file_size_bytes\":" + String.format(Locale.US, "%.3f", avgPartFileSizeBytes)
                    + ",\"dense_double_reference_size_bytes\":" + denseDoubleReferenceSizeBytes
                    + ",\"actual_file_to_dense_double_reference_ratio\":"
                    + String.format(Locale.US, "%.6f", actualFileToDenseDoubleReferenceRatio)
                    + ",\"heap_before_bytes\":" + heapBeforeBytes
                    + ",\"heap_after_bytes\":" + heapAfterBytes
                    + ",\"heap_delta_bytes\":" + heapDeltaBytes
                    + ",\"gc_count_delta\":" + gcCountDelta
                    + ",\"gc_time_delta_ms\":" + gcTimeDeltaMs
                    + ",\"configured_parallel_read_parallelism\":" + configuredParallelReadParallelism
                    + ",\"configured_parallel_write_parallelism\":" + configuredParallelWriteParallelism
                    + ",\"raw_buffer_bytes_property\":" + json(rawBufferBytesProperty)
                    + ",\"keep_files_property\":" + json(keepFilesProperty)
                    + ",\"hdf5_read_parallel_threads_property\":" + json(hdf5ReadParallelThreadsProperty)
                    + ",\"hdf5_read_parallel_min_bytes_property\":" + json(hdf5ReadParallelMinBytesProperty)
                    + ",\"hdf5_read_block_bytes_property\":" + json(hdf5ReadBlockBytesProperty)
                    + ",\"hdf5_read_buffer_bytes_property\":" + json(hdf5ReadBufferBytesProperty)
                    + ",\"hdf5_read_map_bytes_property\":" + json(hdf5ReadMapBytesProperty)
                    + ",\"hdf5_read_mmap_property\":" + json(hdf5ReadMmapProperty)
                    + ",\"hdf5_read_trace_property\":" + json(hdf5ReadTraceProperty)
                    + ",\"hdf5_skip_nnz_property\":" + json(hdf5SkipNnzProperty)
                    + ",\"hdf5_force_dense_property\":" + json(hdf5ForceDenseProperty)
                    + ",\"implementation_path\":" + json(implementationPath)
                    + "}";
        }
    }
}
