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

package org.apache.sysds.test.functions.io;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameReaderParquetParallel;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.runtime.io.FrameWriterParquetParallel;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.junit.Assume;
import org.junit.Test;

/**
 * Manual benchmark for current Parquet frame IO implementations.
 * This benchmark is intentionally disabled by default and must be enabled with:
 * -Dsysds.test.parquet.benchmark=true
 *
 * The benchmark is diagnostic only. It is not a value-level correctness test.
 *
 * Default measured paths:
 * - sequential Parquet frame write/read
 * - current parallel Parquet frame write/read
 * - raw byte streaming read as an IO-only control
 * - Parquet footer-only read as a metadata-only control
 *
 * Default data profiles:
 * - dense_double_only
 * - mixed_schema
 * - sparse_like_double
 *
 * Optional manual multipart experiment:
 * - enable with -Dsysds.test.parquet.manual.parts=2,4,8
 * - this manually creates multiple Parquet part files and benchmarks the current
 *   parallel reader on those multi-file inputs
 *
 * Output:
 * - target/parquet-benchmark.csv
 * - target/parquet-benchmark.json
 *
 * Known limitations:
 * - results are affected by JVM warm-up and OS page cache
 * - heap_before/heap_after are rough diagnostics, not exact allocation counts
 * - raw_io_read measures byte streaming only, not Parquet decoding
 * - manual multipart input is performance-only and not correctness-validated
 */
public class ParquetBenchmarkTest extends AutomatedTestBase {
    private static final String TEST_NAME = "ParquetBenchmarkTest";
    private static final String TEST_DIR = "functions/io/";
    private static final String TEST_CLASS_DIR = TEST_DIR + ParquetBenchmarkTest.class.getSimpleName() + "/";

    private static final String PROFILE_DENSE_DOUBLE = "dense_double_only";
    private static final String PROFILE_MIXED_SCHEMA = "mixed_schema";
    private static final String PROFILE_SPARSE_LIKE_DOUBLE = "sparse_like_double";

    private static volatile long _blackhole = 0;

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
    }

    @Test
    public void benchmarkParquetReadWrite() throws Exception {
        Assume.assumeTrue("Manual benchmark. Enable with -Dsysds.test.parquet.benchmark=true", Boolean.getBoolean("sysds.test.parquet.benchmark"));

        final int rows = Integer.getInteger("sysds.test.parquet.rows", 100_000);
        final int cols = Integer.getInteger("sysds.test.parquet.cols", 50);
        final int warmup = Integer.getInteger("sysds.test.parquet.warmup", 1);
        final int reps = Integer.getInteger("sysds.test.parquet.reps", 3);

        final String[] profiles = parseProfileList(System.getProperty("sysds.test.parquet.profiles", PROFILE_DENSE_DOUBLE + "," + PROFILE_MIXED_SCHEMA + "," + PROFILE_SPARSE_LIKE_DOUBLE));

        final int[] manualParts = parsePositiveIntList(System.getProperty("sysds.test.parquet.manual.parts", ""));

        File csvFile = new File("target/parquet-benchmark.csv");
        File jsonFile = new File("target/parquet-benchmark.json");

        if(csvFile.getParentFile() != null)
            csvFile.getParentFile().mkdirs();

        try(PrintWriter csv = new PrintWriter(new FileWriter(csvFile)); PrintWriter json = new PrintWriter(new FileWriter(jsonFile))) {
            writeCsvHeader(csv);

            json.println("[");
            JsonState jsonState = new JsonState();

            for(String dataProfile : profiles) {
                FrameBlock frameBlock = generateFrame(dataProfile, rows, cols);
                benchmarkProfile(csv, json, jsonState, dataProfile, frameBlock, warmup, reps);

                if(manualParts.length > 0)
                    benchmarkManualMultipart(csv, json, jsonState, dataProfile, frameBlock, manualParts, warmup, reps);
            }

            json.println();
            json.println("]");
        }

        System.out.println("Parquet CSV benchmark results saved under: " + csvFile.getAbsolutePath());
        System.out.println("Parquet JSON benchmark results saved under: " + jsonFile.getAbsolutePath());
    }

    private void benchmarkProfile(
            PrintWriter csv,
            PrintWriter json,
            JsonState jsonState,
            String dataProfile,
            FrameBlock frameBlock,
            int warmup,
            int reps) throws Exception {
        int rows = frameBlock.getNumRows();
        int cols = frameBlock.getNumColumns();
        String profileName = safeName(dataProfile);

        String seqInput = output("parquet_" + profileName + "_seq_input");
        new FrameWriterParquet().writeFrameToHDFS(frameBlock, seqInput, rows, cols);

        String parallelInput = output("parquet_" + profileName + "_parallel_input");
        new FrameWriterParquetParallel().writeFrameToHDFS(frameBlock, parallelInput, rows, cols);

        benchmarkWrite(csv, json, jsonState, dataProfile, "seq", new FrameWriterParquet(), frameBlock, warmup, reps, "");
        benchmarkRawIORead(csv, json, jsonState, dataProfile, "seq", seqInput, frameBlock, warmup, reps, "raw_bytes_only");
        benchmarkFooterRead(csv, json, jsonState, dataProfile, "seq", seqInput, frameBlock, warmup, reps, "parquet_footer_only");
        benchmarkRead(csv, json, jsonState, dataProfile, "seq", new FrameReaderParquet(), seqInput, frameBlock, warmup, reps, "");

        benchmarkWrite(csv, json, jsonState, dataProfile, "parallel", new FrameWriterParquetParallel(), frameBlock, warmup, reps, "current_parallel_write_path");
        benchmarkRawIORead(csv, json, jsonState, dataProfile, "parallel", parallelInput, frameBlock, warmup, reps, "raw_bytes_only");
        benchmarkFooterRead(csv, json, jsonState, dataProfile, "parallel", parallelInput, frameBlock, warmup, reps, "parquet_footer_only");
        benchmarkRead(csv, json, jsonState, dataProfile, "parallel", new FrameReaderParquetParallel(), parallelInput, frameBlock, warmup, reps, "current_parallel_read_path_not_value_validated");
    }

    private void benchmarkManualMultipart(
            PrintWriter csv,
            PrintWriter json,
            JsonState jsonState,
            String dataProfile,
            FrameBlock frameBlock,
            int[] manualParts,
            int warmup,
            int reps) throws Exception {
        for(int numParts : manualParts) {
            if(numParts <= 1 || numParts > frameBlock.getNumRows())
                continue;

            String manualInput = output("parquet_" + safeName(dataProfile) + "_manual_multipart_" + numParts);

            writeManualMultipartInput(frameBlock, manualInput, numParts);

            String impl = "parallel_manual_parts_" + numParts;

            benchmarkRawIORead(csv, json, jsonState, dataProfile, impl, manualInput, frameBlock, warmup, reps, "raw_bytes_only");
            benchmarkFooterRead(csv, json, jsonState, dataProfile, impl, manualInput, frameBlock, warmup, reps, "parquet_footer_only");
            benchmarkRead(csv, json, jsonState, dataProfile, impl, new FrameReaderParquetParallel(), manualInput, frameBlock, warmup, reps, "manual_multipart_input_not_value_validated");
        }
    }

    private void writeCsvHeader(PrintWriter csv) {
        csv.println("data_profile,impl,operation,rows,cols,cells,rep,is_warmup,wall_ms,"
                + "file_size_bytes,num_part_files,avg_part_file_size_bytes,"
                + "hdfs_block_size_bytes,current_writer_estimated_num_part_files,"
                + "current_writer_estimated_num_threads,"
                + "configured_parallel_read_parallelism,configured_parallel_write_parallelism,"
                + "estimated_parallel_read_tasks,"
                + "dense_double_reference_size_bytes,actual_file_to_dense_double_reference_ratio,"
                + "expected_nonzero_fraction,"
                + "reader_value_extraction_mode,frame_materialization,"
                + "heap_before_bytes,heap_after_bytes,heap_delta_bytes,"
                + "gc_count_delta,gc_time_delta_ms,notes");
    }

    private void benchmarkWrite(
            PrintWriter csv,
            PrintWriter json,
            JsonState jsonState,
            String dataProfile,
            String impl,
            FrameWriter writer,
            FrameBlock frameBlock,
            int warmup,
            int reps,
            String notes) throws Exception {
        int rows = frameBlock.getNumRows();
        int cols = frameBlock.getNumColumns();

        for(int i = 0; i < warmup + reps; i++) {
            boolean isWarmup = i < warmup;
            String path = output("parquet_" + safeName(dataProfile) + "_" + impl + "_write_" + i);

            long heapBefore = usedHeap();
            GcStats gcBefore = getGcStats();

            long t0 = System.nanoTime();
            writer.writeFrameToHDFS(frameBlock, path, rows, cols);
            long t1 = System.nanoTime();

            GcStats gcAfter = getGcStats();
            long heapAfter = usedHeap();

            PathStats pathStats = getPathStats(path);

            writeResult(csv, json, jsonState, dataProfile, impl, "write", rows, cols, i, isWarmup, t0, t1, pathStats.fileSizeBytes, pathStats.numPartFiles, heapBefore, heapAfter, gcAfter.count - gcBefore.count, gcAfter.timeMs - gcBefore.timeMs, notes);
        }
    }

    private void benchmarkRead(
            PrintWriter csv,
            PrintWriter json,
            JsonState jsonState,
            String dataProfile,
            String impl,
            FrameReader reader,
            String path,
            FrameBlock reference,
            int warmup,
            int reps,
            String notes) throws Exception {
        int rows = reference.getNumRows();
        int cols = reference.getNumColumns();
        ValueType[] schema = reference.getSchema();
        String[] names = reference.getColumnNames();

        for(int i = 0; i < warmup + reps; i++) {
            boolean isWarmup = i < warmup;

            PathStats pathStats = getPathStats(path);

            long heapBefore = usedHeap();
            GcStats gcBefore = getGcStats();

            long t0 = System.nanoTime();
            FrameBlock ret = reader.readFrameFromHDFS(path, schema, names, rows, cols);
            long t1 = System.nanoTime();

            GcStats gcAfter = getGcStats();
            long heapAfter = usedHeap();

            blackhole(ret);

            writeResult(csv, json, jsonState, dataProfile, impl, "read", rows, cols, i, isWarmup, t0, t1, pathStats.fileSizeBytes, pathStats.numPartFiles, heapBefore, heapAfter, gcAfter.count - gcBefore.count, gcAfter.timeMs - gcBefore.timeMs, notes);
        }
    }

    private void benchmarkRawIORead(
            PrintWriter csv,
            PrintWriter json,
            JsonState jsonState,
            String dataProfile,
            String impl,
            String path,
            FrameBlock reference,
            int warmup,
            int reps,
            String notes) throws Exception {
        int rows = reference.getNumRows();
        int cols = reference.getNumColumns();

        for(int i = 0; i < warmup + reps; i++) {
            boolean isWarmup = i < warmup;

            PathStats pathStats = getPathStats(path);

            long heapBefore = usedHeap();
            GcStats gcBefore = getGcStats();

            long t0 = System.nanoTime();
            long bytesRead = readRawBytes(path);
            long t1 = System.nanoTime();

            GcStats gcAfter = getGcStats();
            long heapAfter = usedHeap();

            blackhole(bytesRead);

            writeResult(csv, json, jsonState, dataProfile, impl, "raw_io_read", rows, cols, i, isWarmup, t0, t1, pathStats.fileSizeBytes, pathStats.numPartFiles, heapBefore, heapAfter, gcAfter.count - gcBefore.count, gcAfter.timeMs - gcBefore.timeMs, notes);
        }
    }

    private void benchmarkFooterRead(
            PrintWriter csv,
            PrintWriter json,
            JsonState jsonState,
            String dataProfile,
            String impl,
            String path,
            FrameBlock reference,
            int warmup,
            int reps,
            String notes) throws Exception {
        int rows = reference.getNumRows();
        int cols = reference.getNumColumns();

        for(int i = 0; i < warmup + reps; i++) {
            boolean isWarmup = i < warmup;

            PathStats pathStats = getPathStats(path);

            long heapBefore = usedHeap();
            GcStats gcBefore = getGcStats();

            long t0 = System.nanoTime();
            long footerInfo = readParquetFooters(path);
            long t1 = System.nanoTime();

            GcStats gcAfter = getGcStats();
            long heapAfter = usedHeap();

            blackhole(footerInfo);

            writeResult(csv, json, jsonState, dataProfile, impl, "footer_read", rows, cols, i, isWarmup, t0, t1, pathStats.fileSizeBytes, pathStats.numPartFiles, heapBefore, heapAfter, gcAfter.count - gcBefore.count, gcAfter.timeMs - gcBefore.timeMs, notes);
        }
    }

    private void writeManualMultipartInput(FrameBlock frameBlock, String dirName, int numParts) throws Exception {
        Configuration conf = ConfigurationManager.getCachedJobConf();
        Path dir = new Path(dirName);
        FileSystem fs = dir.getFileSystem(conf);

        if(fs.exists(dir))
            fs.delete(dir, true);

        fs.mkdirs(dir);

        FrameWriter writer = new FrameWriterParquet();

        int rows = frameBlock.getNumRows();
        int cols = frameBlock.getNumColumns();
        int chunkSize = (int) Math.ceil((double) rows / numParts);

        for(int part = 0; part < numParts; part++) {
            int startRow = part * chunkSize;
            int endRow = Math.min((part + 1) * chunkSize, rows);

            if(startRow >= endRow)
                continue;

            FrameBlock slice = frameBlock.slice(startRow, endRow - 1);
            Path partPath = new Path(dir, getManualPartFileName(part));

            writer.writeFrameToHDFS(slice, partPath.toString(), slice.getNumRows(), cols);
        }
    }

    private String getManualPartFileName(int part) {
        return String.format(Locale.US, "part-%05d", part);
    }

    private void writeResult(
            PrintWriter csv,
            PrintWriter json,
            JsonState jsonState,
            String dataProfile,
            String impl,
            String operation,
            int rows,
            int cols,
            int rep,
            boolean isWarmup,
            long t0,
            long t1,
            long fileSizeBytes,
            int numPartFiles,
            long heapBefore,
            long heapAfter,
            long gcCountDelta,
            long gcTimeDeltaMs,
            String notes) {
        double wallMs = (t1 - t0) / 1e6;
        BenchmarkDiagnostics diag = getBenchmarkDiagnostics(dataProfile, impl, operation, rows, cols, fileSizeBytes, numPartFiles);

        csv.printf(Locale.US,
                "%s,%s,%s,%d,%d,%d,%d,%s,%.3f,%d,%d,%.3f,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%s,%s,%d,%d,%d,%d,%d,%s%n",
                escapeCsv(dataProfile),
                escapeCsv(impl),
                escapeCsv(operation),
                rows,
                cols,
                diag.cells,
                rep,
                isWarmup,
                wallMs,
                fileSizeBytes,
                numPartFiles,
                diag.avgPartFileSizeBytes,
                diag.hdfsBlockSizeBytes,
                diag.currentWriterEstimatedNumPartFiles,
                diag.currentWriterEstimatedNumThreads,
                diag.configuredParallelReadParallelism,
                diag.configuredParallelWriteParallelism,
                diag.estimatedParallelReadTasks,
                diag.denseDoubleReferenceSizeBytes,
                diag.actualFileToDenseDoubleReferenceRatio,
                diag.expectedNonZeroFraction,
                escapeCsv(diag.readerValueExtractionMode),
                escapeCsv(diag.frameMaterialization),
                heapBefore,
                heapAfter,
                heapAfter - heapBefore,
                gcCountDelta,
                gcTimeDeltaMs,
                escapeCsv(notes));
        csv.flush();

        if(jsonState.hasEntries)
            json.println(",");
        else
            jsonState.hasEntries = true;

        json.printf(Locale.US,
                "  {\"data_profile\":\"%s\",\"impl\":\"%s\",\"operation\":\"%s\","
                        + "\"rows\":%d,\"cols\":%d,\"cells\":%d,"
                        + "\"rep\":%d,\"is_warmup\":%s,\"wall_ms\":%.3f,"
                        + "\"file_size_bytes\":%d,\"num_part_files\":%d,"
                        + "\"avg_part_file_size_bytes\":%.3f,"
                        + "\"hdfs_block_size_bytes\":%d,"
                        + "\"current_writer_estimated_num_part_files\":%d,"
                        + "\"current_writer_estimated_num_threads\":%d,"
                        + "\"configured_parallel_read_parallelism\":%d,"
                        + "\"configured_parallel_write_parallelism\":%d,"
                        + "\"estimated_parallel_read_tasks\":%d,"
                        + "\"dense_double_reference_size_bytes\":%d,"
                        + "\"actual_file_to_dense_double_reference_ratio\":%.6f,"
                        + "\"expected_nonzero_fraction\":%.6f,"
                        + "\"reader_value_extraction_mode\":\"%s\","
                        + "\"frame_materialization\":\"%s\","
                        + "\"heap_before_bytes\":%d,\"heap_after_bytes\":%d,"
                        + "\"heap_delta_bytes\":%d,"
                        + "\"gc_count_delta\":%d,\"gc_time_delta_ms\":%d,"
                        + "\"notes\":\"%s\"}",
                escapeJson(dataProfile),
                escapeJson(impl),
                escapeJson(operation),
                rows,
                cols,
                diag.cells,
                rep,
                isWarmup,
                wallMs,
                fileSizeBytes,
                numPartFiles,
                diag.avgPartFileSizeBytes,
                diag.hdfsBlockSizeBytes,
                diag.currentWriterEstimatedNumPartFiles,
                diag.currentWriterEstimatedNumThreads,
                diag.configuredParallelReadParallelism,
                diag.configuredParallelWriteParallelism,
                diag.estimatedParallelReadTasks,
                diag.denseDoubleReferenceSizeBytes,
                diag.actualFileToDenseDoubleReferenceRatio,
                diag.expectedNonZeroFraction,
                escapeJson(diag.readerValueExtractionMode),
                escapeJson(diag.frameMaterialization),
                heapBefore,
                heapAfter,
                heapAfter - heapBefore,
                gcCountDelta,
                gcTimeDeltaMs,
                escapeJson(notes));
        json.flush();
    }

    private BenchmarkDiagnostics getBenchmarkDiagnostics(
            String dataProfile,
            String impl,
            String operation,
            int rows,
            int cols,
            long fileSizeBytes,
            int numPartFiles) {
        long cells = (long) rows * cols;
        long hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();

        int configuredParallelReadParallelism = OptimizerUtils.getParallelBinaryReadParallelism();
        int configuredParallelWriteParallelism = OptimizerUtils.getParallelBinaryWriteParallelism();

        long currentWriterEstimatedNumPartFiles = hdfsBlockSize > 0 ? Math.max(cells / hdfsBlockSize, 1) : 1;
        int currentWriterEstimatedNumThreads = (int) Math.min(configuredParallelWriteParallelism, currentWriterEstimatedNumPartFiles);

        int estimatedParallelReadTasks = estimateParallelReadTasks(impl, operation, numPartFiles, configuredParallelReadParallelism);

        long denseDoubleReferenceSizeBytes = cells * 8;
        double avgPartFileSizeBytes = numPartFiles > 0 ? ((double) fileSizeBytes / numPartFiles) : -1.0;
        double actualFileToDenseDoubleReferenceRatio = denseDoubleReferenceSizeBytes > 0 ? ((double) fileSizeBytes / denseDoubleReferenceSizeBytes) : -1.0;

        String readerValueExtractionMode = getReaderValueExtractionMode(impl, operation);
        String frameMaterialization = "FrameBlock";
        double expectedNonZeroFraction = getExpectedNonZeroFraction(dataProfile);

        return new BenchmarkDiagnostics(cells, hdfsBlockSize, currentWriterEstimatedNumPartFiles, currentWriterEstimatedNumThreads, configuredParallelReadParallelism, configuredParallelWriteParallelism, estimatedParallelReadTasks, denseDoubleReferenceSizeBytes, avgPartFileSizeBytes, actualFileToDenseDoubleReferenceRatio, expectedNonZeroFraction, readerValueExtractionMode, frameMaterialization);
    }

    private int estimateParallelReadTasks(
            String impl,
            String operation,
            int numPartFiles,
            int configuredParallelReadParallelism) {
        if(!"read".equals(operation))
            return 0;

        if(!impl.startsWith("parallel"))
            return 1;

        if(numPartFiles <= 0)
            return 0;

        return Math.min(configuredParallelReadParallelism, numPartFiles);
    }

    private String getReaderValueExtractionMode(String impl, String operation) {
        if(!"read".equals(operation))
            return "not_applicable";

        if("seq".equals(impl))
            return "typed_getters";

        if(impl.startsWith("parallel"))
            return "getValueToString";

        return "unknown";
    }

    private double getExpectedNonZeroFraction(String dataProfile) {
        if(PROFILE_DENSE_DOUBLE.equals(dataProfile))
            return 1.0;
        else if(PROFILE_SPARSE_LIKE_DOUBLE.equals(dataProfile))
            return 0.05;
        else
            return -1.0;
    }

    private FrameBlock generateFrame(String dataProfile, int rows, int cols) {
        if(PROFILE_DENSE_DOUBLE.equals(dataProfile))
            return generateDenseDoubleFrame(rows, cols);
        else if(PROFILE_MIXED_SCHEMA.equals(dataProfile))
            return generateMixedSchemaFrame(rows, cols);
        else if(PROFILE_SPARSE_LIKE_DOUBLE.equals(dataProfile))
            return generateSparseLikeDoubleFrame(rows, cols);
        else
            throw new RuntimeException("Unknown Parquet benchmark data profile: " + dataProfile);
    }

    private FrameBlock generateDenseDoubleFrame(int rows, int cols) {
        ValueType[] schema = new ValueType[cols];
        for(int j = 0; j < cols; j++)
            schema[j] = ValueType.FP64;

        FrameBlock frameBlock = new FrameBlock(schema);

        for(int i = 0; i < rows; i++) {
            Object[] row = new Object[cols];
            for(int j = 0; j < cols; j++)
                row[j] = (double) (i * cols + j + 1);
            frameBlock.appendRow(row);
        }

        return frameBlock;
    }

    private FrameBlock generateSparseLikeDoubleFrame(int rows, int cols) {
        ValueType[] schema = new ValueType[cols];
        for(int j = 0; j < cols; j++)
            schema[j] = ValueType.FP64;

        FrameBlock frameBlock = new FrameBlock(schema);

        for(int i = 0; i < rows; i++) {
            Object[] row = new Object[cols];
            for(int j = 0; j < cols; j++) {
                int linearIndex = i * cols + j;
                if(linearIndex % 20 == 0)
                    row[j] = (double) (linearIndex + 1);
                else
                    row[j] = 0.0;
            }
            frameBlock.appendRow(row);
        }

        return frameBlock;
    }

    private FrameBlock generateMixedSchemaFrame(int rows, int cols) {
        ValueType[] schema = new ValueType[cols];
        for(int j = 0; j < cols; j++) {
            switch(j % 5) {
                case 0:
                    schema[j] = ValueType.FP64;
                    break;
                case 1:
                    schema[j] = ValueType.INT64;
                    break;
                case 2:
                    schema[j] = ValueType.BOOLEAN;
                    break;
                case 3:
                    schema[j] = ValueType.STRING;
                    break;
                default:
                    schema[j] = ValueType.INT32;
            }
        }

        FrameBlock frameBlock = new FrameBlock(schema);

        for(int i = 0; i < rows; i++) {
            Object[] row = new Object[cols];
            for(int j = 0; j < cols; j++) {
                switch(schema[j]) {
                    case FP64:
                        row[j] = (double) (i * cols + j + 1);
                        break;
                    case INT64:
                        row[j] = (long) (i * cols + j + 1);
                        break;
                    case BOOLEAN:
                        row[j] = ((i + j) % 2 == 0);
                        break;
                    case STRING:
                        row[j] = "s_" + (i % 1000) + "_" + j;
                        break;
                    case INT32:
                        row[j] = i + j;
                        break;
                    default:
                        throw new RuntimeException("Unsupported generated type: " + schema[j]);
                }
            }
            frameBlock.appendRow(row);
        }

        return frameBlock;
    }

    private long readRawBytes(String fname) throws IOException {
        Configuration conf = ConfigurationManager.getCachedJobConf();
        List<Path> files = listDataFiles(fname, conf);

        byte[] buffer = new byte[1024 * 1024];
        long total = 0;

        for(Path file : files) {
            FileSystem fs = file.getFileSystem(conf);

            try(FSDataInputStream in = fs.open(file)) {
                int n;
                while((n = in.read(buffer)) != -1)
                    total += n;
            }
        }

        return total;
    }

    private long readParquetFooters(String fname) throws IOException {
        Configuration conf = ConfigurationManager.getCachedJobConf();
        List<Path> files = listDataFiles(fname, conf);

        long checksum = 0;

        for(Path file : files) {
            try(ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(file, conf))) {
                ParquetMetadata metadata = reader.getFooter();

                checksum += metadata.getBlocks().size();
                checksum += metadata.getFileMetaData().getSchema().getFieldCount();

                for(int i = 0; i < metadata.getBlocks().size(); i++)
                    checksum += metadata.getBlocks().get(i).getRowCount();
            }
        }

        return checksum;
    }

    private List<Path> listDataFiles(String fname, Configuration conf) throws IOException {
        Path path = new Path(fname);
        FileSystem fileSystem = path.getFileSystem(conf);

        List<Path> files = new ArrayList<>();

        if(!fileSystem.exists(path))
            return files;

        FileStatus status = fileSystem.getFileStatus(path);

        if(status.isFile()) {
            files.add(path);
            return files;
        }

        for(FileStatus child : fileSystem.listStatus(path)) {
            if(child.isFile() && isDataFile(child.getPath()))
                files.add(child.getPath());
        }

        return files;
    }

    private boolean isDataFile(Path path) {
        String name = path.getName();

        return !name.startsWith("_")
                && !name.startsWith(".")
                && !name.endsWith(".crc");
    }

    private PathStats getPathStats(String fname) {
        try {
            Configuration conf = ConfigurationManager.getCachedJobConf();
            List<Path> files = listDataFiles(fname, conf);

            long totalSize = 0;
            int numFiles = 0;

            for(Path file : files) {
                FileSystem fileSystem = file.getFileSystem(conf);
                FileStatus status = fileSystem.getFileStatus(file);

                if(status.isFile()) {
                    totalSize += status.getLen();
                    numFiles++;
                }
            }

            return new PathStats(totalSize, numFiles);
        }
        catch(Exception ex) {
            return new PathStats(-1, -1);
        }
    }

    private String[] parseProfileList(String value) {
        if(value == null || value.trim().isEmpty())
            return new String[] {PROFILE_DENSE_DOUBLE, PROFILE_MIXED_SCHEMA, PROFILE_SPARSE_LIKE_DOUBLE};

        String[] tokens = value.split(",");
        List<String> profiles = new ArrayList<>();

        for(String token : tokens) {
            String profile = token.trim();

            if(profile.isEmpty())
                continue;

            if(!PROFILE_DENSE_DOUBLE.equals(profile) && !PROFILE_MIXED_SCHEMA.equals(profile)
                    && !PROFILE_SPARSE_LIKE_DOUBLE.equals(profile)) {
                throw new RuntimeException("Unknown Parquet benchmark data profile: " + profile);
            }

            profiles.add(profile);
        }

        if(profiles.isEmpty())
            throw new RuntimeException("No valid Parquet benchmark data profiles specified.");

        return profiles.toArray(new String[0]);
    }

    private int[] parsePositiveIntList(String value) {
        if(value == null || value.trim().isEmpty())
            return new int[0];

        String[] tokens = value.split(",");
        int[] tmp = new int[tokens.length];
        int count = 0;

        for(String token : tokens) {
            String trimmed = token.trim();

            if(trimmed.isEmpty())
                continue;

            int v = Integer.parseInt(trimmed);

            if(v > 0)
                tmp[count++] = v;
        }

        int[] ret = new int[count];
        System.arraycopy(tmp, 0, ret, 0, count);
        return ret;
    }

    private String safeName(String s) {
        return s.replaceAll("[^A-Za-z0-9_\\-]", "_");
    }

    private long usedHeap() {
        Runtime rt = Runtime.getRuntime();
        return rt.totalMemory() - rt.freeMemory();
    }

    private GcStats getGcStats() {
        long count = 0;
        long timeMs = 0;

        for(GarbageCollectorMXBean bean : ManagementFactory.getGarbageCollectorMXBeans()) {
            long c = bean.getCollectionCount();
            long t = bean.getCollectionTime();

            if(c >= 0)
                count += c;
            if(t >= 0)
                timeMs += t;
        }

        return new GcStats(count, timeMs);
    }

    private void blackhole(FrameBlock frameBlock) {
        if(frameBlock == null)
            throw new RuntimeException("Unexpected null FrameBlock.");

        _blackhole ^= frameBlock.getNumRows();
        _blackhole ^= frameBlock.getNumColumns();
    }

    private void blackhole(long value) {
        _blackhole ^= value;
    }

    private String escapeCsv(String s) {
        if(s == null)
            return "";

        boolean needsQuotes =
                s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0 || s.indexOf('\r') >= 0;

        if(!needsQuotes)
            return s;

        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    private String escapeJson(String s) {
        if(s == null)
            return "";

        StringBuilder stringBuilder = new StringBuilder();
        for(int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch(c) {
                case '"':
                    stringBuilder.append("\\\"");
                    break;
                case '\\':
                    stringBuilder.append("\\\\");
                    break;
                case '\b':
                    stringBuilder.append("\\b");
                    break;
                case '\f':
                    stringBuilder.append("\\f");
                    break;
                case '\n':
                    stringBuilder.append("\\n");
                    break;
                case '\r':
                    stringBuilder.append("\\r");
                    break;
                case '\t':
                    stringBuilder.append("\\t");
                    break;
                default:
                    if(c < 0x20)
                        stringBuilder.append(String.format("\\u%04x", (int) c));
                    else
                        stringBuilder.append(c);
            }
        }
        return stringBuilder.toString();
    }

    private static class JsonState {
        private boolean hasEntries = false;
    }

    private static class GcStats {
        private final long count;
        private final long timeMs;

        private GcStats(long count, long timeMs) {
            this.count = count;
            this.timeMs = timeMs;
        }
    }

    private static class PathStats {
        private final long fileSizeBytes;
        private final int numPartFiles;

        private PathStats(long fileSizeBytes, int numPartFiles) {
            this.fileSizeBytes = fileSizeBytes;
            this.numPartFiles = numPartFiles;
        }
    }

    private static class BenchmarkDiagnostics {
        private final long cells;
        private final long hdfsBlockSizeBytes;
        private final long currentWriterEstimatedNumPartFiles;
        private final int currentWriterEstimatedNumThreads;
        private final int configuredParallelReadParallelism;
        private final int configuredParallelWriteParallelism;
        private final int estimatedParallelReadTasks;
        private final long denseDoubleReferenceSizeBytes;
        private final double avgPartFileSizeBytes;
        private final double actualFileToDenseDoubleReferenceRatio;
        private final double expectedNonZeroFraction;
        private final String readerValueExtractionMode;
        private final String frameMaterialization;

        private BenchmarkDiagnostics(
                long cells,
                long hdfsBlockSizeBytes,
                long currentWriterEstimatedNumPartFiles,
                int currentWriterEstimatedNumThreads,
                int configuredParallelReadParallelism,
                int configuredParallelWriteParallelism,
                int estimatedParallelReadTasks,
                long denseDoubleReferenceSizeBytes,
                double avgPartFileSizeBytes,
                double actualFileToDenseDoubleReferenceRatio,
                double expectedNonZeroFraction,
                String readerValueExtractionMode,
                String frameMaterialization) {
            this.cells = cells;
            this.hdfsBlockSizeBytes = hdfsBlockSizeBytes;
            this.currentWriterEstimatedNumPartFiles = currentWriterEstimatedNumPartFiles;
            this.currentWriterEstimatedNumThreads = currentWriterEstimatedNumThreads;
            this.configuredParallelReadParallelism = configuredParallelReadParallelism;
            this.configuredParallelWriteParallelism = configuredParallelWriteParallelism;
            this.estimatedParallelReadTasks = estimatedParallelReadTasks;
            this.denseDoubleReferenceSizeBytes = denseDoubleReferenceSizeBytes;
            this.avgPartFileSizeBytes = avgPartFileSizeBytes;
            this.actualFileToDenseDoubleReferenceRatio = actualFileToDenseDoubleReferenceRatio;
            this.expectedNonZeroFraction = expectedNonZeroFraction;
            this.readerValueExtractionMode = readerValueExtractionMode;
            this.frameMaterialization = frameMaterialization;
        }
    }
}