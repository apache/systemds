package org.apache.sysds.runtime.io;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.cog.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;

public class ReaderCOGParallel extends MatrixReader{
    protected final FileFormatPropertiesCOG _props;
    final private int _numThreads;

    public ReaderCOGParallel(FileFormatPropertiesCOG props) {
        _props = props;
        _numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
    }
    @Override
    public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {
        JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
        Path path = new Path(fname);
        FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

        BufferedInputStream bis = new BufferedInputStream(fs.open(path));
        return readCOG(bis, estnnz);
    }

    @Override
    public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {
        BufferedInputStream bis = new BufferedInputStream(is);
        return readCOG(bis, estnnz);
    }

    /**
     * Reads a COG file from a BufferedInputStream.
     * Not handling number of columns or rows, as this can be inferred from the data, but
     * may be used in the future for validation or possibly removed as a requirement for COG.
     * Specific to COG files, normal TIFFs will break because they aren't tiled, only
     * tiled data is supported.
     * @param bis
     * @return
     */
    private MatrixBlock readCOG(BufferedInputStream bis, long estnnz) throws IOException {
        COGByteReader byteReader = new COGByteReader(bis);
        COGHeader cogHeader = COGHeader.readCOGHeader(byteReader);

        // Check compatibility of the file with our reader
        // Certain options are not supported, and we need to filter out some non-standard options
        String isCompatible = COGHeader.isCompatible(cogHeader.getIFD());
        if (!isCompatible.equals("")) {
            throw new DMLRuntimeException("Incompatible COG file: " + isCompatible);
        }

        // TODO: Currently only reads the first image which is the full resolution image
        // In the future, this could be extended to read the overviews as well

        // Prepare everything for reading the actual image data
        int rows = -1;
        int cols = -1;
        int bands = -1;
        int[] bitsPerSample = null;
        SampleFormatDataTypes[] sampleFormat = null;
        int planarConfiguration = -1;
        int tileWidth = -1;
        int tileLength = -1;
        int[] tileOffsets = null;
        int[] bytesPerTile = null;
        int compression = -1;

        // Set the attributes correctly from the IFD tags
        for (IFDTag ifd : cogHeader.getIFD()) {
            IFDTagDictionary tag = ifd.getTagId();
            switch (tag) {
                case ImageWidth:
                    cols = ifd.getData()[0].intValue();
                    break;
                case ImageLength:
                    rows = ifd.getData()[0].intValue();
                    break;
                case SamplesPerPixel:
                    bands = ifd.getData()[0].intValue();
                    break;
                case BitsPerSample:
                    bitsPerSample = Arrays.stream(ifd.getData()).mapToInt(Number::intValue).toArray();
                    break;
                case TileWidth:
                    tileWidth = ifd.getData()[0].intValue();
                    break;
                case TileLength:
                    tileLength = ifd.getData()[0].intValue();
                    break;
                case TileOffsets:
                    tileOffsets = Arrays.stream(ifd.getData()).mapToInt(Number::intValue).toArray();
                    break;
                case TileByteCounts:
                    if (ifd.getData() != null) {
                        bytesPerTile = Arrays.stream(ifd.getData()).mapToInt(Number::intValue).toArray();
                    } else {
                        bytesPerTile = new int[tileOffsets.length];
                        for (int tile = 0; tile < tileOffsets.length; tile++) {
                            int bits = 0;
                            for (int band = 0; band < bands; band++) {
                                bits += bitsPerSample[band];
                            }
                            bytesPerTile[tile] = tileWidth * tileLength * (bits / 8);
                        }
                    }
                    break;
                case SampleFormat:
                    int dataCount = ifd.getDataCount();
                    sampleFormat = new SampleFormatDataTypes[dataCount];
                    for (int i = 0; i < dataCount; i++) {
                        sampleFormat[i] = SampleFormatDataTypes.valueOf(ifd.getData()[i].intValue());
                    }
                    break;
                case PlanarConfiguration:
                    planarConfiguration = ifd.getData()[0].intValue();
                    break;
                case Compression:
                    compression = ifd.getData()[0].intValue();
                    break;
            }
        }

        // number of tiles for Width and Length
        int tileCols = cols / tileWidth;
        int tileRows = rows / tileLength;

        // total number of tiles if every tile contains all bands
        int calculatedAmountTiles = tileCols * tileRows;
        // actual given number of tiles, longer for PlanarConfiguration=2
        int actualAmountTiles = tileOffsets.length;

        int currentTileCol = 0;
        int currentTileRow = 0;
        int currentBand = 0;

        ExecutorService pool = CommonThreadPool.get(_numThreads);

        MatrixBlock outputMatrix = createOutputMatrixBlock(rows, cols * bands, rows, estnnz, true, true);

        try {
            ArrayList<Callable<MatrixBlock>> tasks = new ArrayList<>();
            // Principle: We're reading all tiles in sequence as I/O likely won't benefit from parallel reads
            // in most cases.
            // Then: Process the read tile data in parallel
            for (int currenTileIdx = 0; currenTileIdx < actualAmountTiles; currenTileIdx++) {
                // First read the bytes for the new tile
                int bytesToRead = (tileOffsets[currenTileIdx] - byteReader.getTotalBytesRead()) + bytesPerTile[currenTileIdx];
                byteReader.mark(bytesToRead);
                byteReader.readBytes(tileOffsets[currenTileIdx] - byteReader.getTotalBytesRead());
                byte[] currentTileData = byteReader.readBytes(bytesPerTile[currenTileIdx]);

                byteReader.reset();

                if (compression == 8) {
                    currentTileData = COGCompressionUtils.decompressDeflate(currentTileData);
                }

                TileProcessor tileProcessor;
                if (planarConfiguration == 1) {
                    // Every band is in the same tile, e.g. RGBRGBRGB
                     tileProcessor = new TileProcessor(cols * bands, currentTileData, currentTileRow, currentTileCol,
                            tileWidth, tileLength, bands, bitsPerSample, sampleFormat, cogHeader, outputMatrix,
                            planarConfiguration, _numThreads<=actualAmountTiles);

                     currentTileCol++;
                    if (currentTileCol >= tileCols) {
                        currentTileCol = 0;
                        currentTileRow++;
                    }
                } else if (planarConfiguration == 2) {
                    // Every band is in a different tile, e.g. RRRGGGBBB
                    // Note here that first all tiles from a single band are present
                    // after that all tiles from the next band are present and so on (so they don't interleave)
                    if (currenTileIdx - (currentBand * calculatedAmountTiles) >= calculatedAmountTiles) {
                        currentTileCol = 0;
                        currentTileRow = 0;
                        currentBand++;
                    }

                    tileProcessor = new TileProcessor(cols * bands, currentTileData, currentTileRow, currentTileCol,
                            tileWidth, tileLength, bands, bitsPerSample, sampleFormat, cogHeader, outputMatrix,
                            planarConfiguration, _numThreads<=actualAmountTiles, currentBand);

                    currentTileCol++;

                    if (currentTileCol >= tileCols) {
                        currentTileCol = 0;
                        currentTileRow++;
                    }
                } else {
                    throw new DMLRuntimeException("Unsupported Planar Configuration: " + planarConfiguration);
                }
                tasks.add(tileProcessor);
            }

            try {
                for (Future<MatrixBlock> result : pool.invokeAll(tasks)) {
                    result.get();
                }

                if (outputMatrix.isInSparseFormat() && tileWidth < cols) {
                    sortSparseRowsParallel(outputMatrix, rows, _numThreads, pool);
                }
            } catch (Exception e) {
                throw new IOException("Error during parallel task execution.", e);
            }

        } catch (IOException e) {
            throw new IOException("Thread pool issue or file reading error.", e);
        } finally {
            pool.shutdown();
        }

        // TODO: If the tile is compressed, decompress the currentTileData here

        outputMatrix.examSparsity();
        return outputMatrix;
    }

    public class TileProcessor implements Callable<MatrixBlock> {

        private final int clen;
        private final byte[] tileData;
        private final int tileRow;
        private final int tileCol;
        private final int tileWidth;
        private final int tileLength;
        private final int bands;
        private final int[] bitsPerSample;
        private final SampleFormatDataTypes[] sampleFormat;
        private final COGHeader cogHeader;
        private final MatrixBlock _dest;
        private final int planarConfiguration;
        private final boolean sparse;
        private final boolean sync;
        private final int band;

        public TileProcessor(int clen, byte[] tileData, int tileRow, int tileCol, int tileWidth, int tileLength,
                             int bands, int[] bitsPerSample, SampleFormatDataTypes[] sampleFormat, COGHeader cogHeader,
                             MatrixBlock dest, int planarConfiguration, boolean sync) {
            this.clen = clen;
            this.tileData = tileData;
            this.tileRow = tileRow;
            this.tileCol = tileCol;
            this.tileWidth = tileWidth;
            this.tileLength = tileLength;
            this.bands = bands;
            this.bitsPerSample = bitsPerSample;
            this.sampleFormat = sampleFormat;
            this.cogHeader = cogHeader;
            this._dest = dest;
            this.planarConfiguration = planarConfiguration;
            this.sparse = _dest.isInSparseFormat();
            this.sync = sync;
            this.band = 0;
        }

        public TileProcessor(int clen, byte[] tileData, int tileRow, int tileCol, int tileWidth, int tileLength,
                             int bands, int[] bitsPerSample, SampleFormatDataTypes[] sampleFormat, COGHeader cogHeader,
                             MatrixBlock dest, int planarConfiguration, boolean sync, int band) {
            this.clen = clen;
            this.tileData = tileData;
            this.tileRow = tileRow;
            this.tileCol = tileCol;
            this.tileWidth = tileWidth;
            this.tileLength = tileLength;
            this.bands = bands;
            this.bitsPerSample = bitsPerSample;
            this.sampleFormat = sampleFormat;
            this.cogHeader = cogHeader;
            this._dest = dest;
            this.planarConfiguration = planarConfiguration;
            this.sparse = _dest.isInSparseFormat();
            this.sync = sync;
            this.band = band;
        }

        @Override
        public MatrixBlock call() throws Exception {
            if (planarConfiguration==1) {
                processTileByPixel();
            }
            else if (planarConfiguration==2){ // && calculatedAmountTiles * bands == tileOffsets.length){
                processTileByBand();
            }
            else{
                throw new DMLRuntimeException("Unsupported Planar Configuration: " + planarConfiguration);
            }
            return _dest;
        }

        private void processTileByPixel() {
            int pixelsRead = 0;
            int bytesRead = 0;
            int currentRow = 0;

            MatrixBlock tileMatrix = new MatrixBlock(tileLength, tileWidth*bands, sparse);

            if( sparse ) {
                tileMatrix.allocateAndResetSparseBlock(true, SparseBlock.Type.CSR);
                tileMatrix.getSparseBlock().allocate(0,  tileLength*tileWidth*bands);
            }

            while (currentRow < tileLength && pixelsRead < tileWidth) {
                for (int bandIdx = 0; bandIdx < bands; bandIdx++) {
                    double value = 0;
                    int sampleLength = bitsPerSample[bandIdx] / 8;

                    switch (sampleFormat[bandIdx]) {
                        case UNSIGNED_INTEGER:
                        case UNDEFINED:
                            value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, false, false, false).doubleValue();
                            break;
                        case SIGNED_INTEGER:
                            value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, false, true, false).doubleValue();
                            break;
                        case FLOATING_POINT:
                            value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, true, false, false).doubleValue();
                            break;
                    }

                    bytesRead += sampleLength;
                    tileMatrix.set(currentRow, (pixelsRead * bands) + bandIdx, value);
                }

                pixelsRead++;
                if (pixelsRead >= tileWidth) {
                    pixelsRead = 0;
                    currentRow++;
                }
            }

            try {
                int rowOffset = tileRow * tileLength;
                int colOffset = tileCol * tileWidth * bands;
                if (sparse) {
                    // if outputMatrix is sparse apply synchronisation if tiles are more narrow then outputMatrix
                    SparseBlock sblock = _dest.getSparseBlock();
                    if (tileWidth < clen) {
                        if (sblock instanceof SparseBlockMCSR && sblock.get(rowOffset) != null) {
                            for (int i = 0; i < tileLength; i++)
                                synchronized (sblock.get(rowOffset + i)) {
                                _dest.appendRowToSparse(sblock, tileMatrix, i,
                                        rowOffset,
                                        colOffset, true);
                                }
                        }
                        else{
                            synchronized (_dest) {
                                _dest.appendToSparse(
                                        tileMatrix,
                                        rowOffset,
                                        colOffset);
                            }
                        }
                    }
                    else {
                        _dest.appendToSparse(
                                tileMatrix,
                                rowOffset,
                                colOffset);
                    }
                }
                else {
                    _dest.copy(rowOffset, rowOffset + tileLength - 1,
                            colOffset, colOffset + (tileWidth * bands) -1,
                            tileMatrix, false);
                }
            } catch (RuntimeException e) {
                throw new DMLRuntimeException("Error while processing tile", e);
            }
        }




        private void processTileByBand() {
            int pixelsRead = 0;
            int bytesRead = 0;
            int currentRow = 0;

            MatrixBlock tileMatrix = new MatrixBlock(tileLength, tileWidth*bands, sparse);

            if( sparse ) {
                tileMatrix.allocateAndResetSparseBlock(true, SparseBlock.Type.CSR);
                tileMatrix.getSparseBlock().allocate(0,  tileLength*tileWidth*bands);
            }

            while (currentRow < tileLength && pixelsRead < tileWidth) {
                double value = 0;
                int sampleLength = bitsPerSample[band] / 8;

                switch (sampleFormat[band]) {
                    case UNSIGNED_INTEGER:
                    case UNDEFINED:
                        value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, false, false, false).doubleValue();
                        break;
                    case SIGNED_INTEGER:
                        value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, false, true, false).doubleValue();
                        break;
                    case FLOATING_POINT:
                        value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, true, false, false).doubleValue();
                        break;
                }

                bytesRead += sampleLength;
                tileMatrix.set(currentRow, (pixelsRead * bands) + band, value);

                pixelsRead++;
                if (pixelsRead >= tileWidth) {
                    pixelsRead = 0;
                    currentRow++;
                }
            }

            try {
                int rowOffset = tileRow * tileLength;
                int colOffset = tileCol * tileWidth * bands;
                if (sparse) {
                    // if outputMatrix is sparse apply synchronisation if tiles are more narrow then outputMatrix
                    SparseBlock sblock = _dest.getSparseBlock();
                    if (tileWidth < clen) {
                        if (sblock instanceof SparseBlockMCSR && sblock.get(rowOffset) != null) {
                            for (int i = 0; i < tileLength; i++)
                                synchronized (sblock.get(rowOffset + i)) {
                                    _dest.appendRowToSparse(sblock, tileMatrix, i,
                                            rowOffset,
                                            colOffset, true);
                                }
                        }
                        else{
                            synchronized (_dest) {
                                _dest.appendToSparse(
                                        tileMatrix,
                                        rowOffset,
                                        colOffset);
                            }
                        }
                    }
                    else {
                        _dest.appendToSparse(
                                tileMatrix,
                                rowOffset,
                                colOffset);
                    }
                }
                else {
                    _dest.copy(rowOffset, rowOffset + tileLength - 1,
                            colOffset, colOffset + (tileWidth * bands) -1,
                            tileMatrix, false);
                }
            } catch (RuntimeException e) {
                throw new DMLRuntimeException("Error while processing tile", e);
            }
        }

    }
}
