package org.apache.sysds.runtime.io;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.cog.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.zip.DataFormatException;

public class ReaderCOG extends MatrixReader{
    protected final FileFormatPropertiesCOG _props;

    public ReaderCOG(FileFormatPropertiesCOG props) {
        _props = props;
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

        MatrixBlock outputMatrix = createOutputMatrixBlock(rows, cols * bands, rows, estnnz, true, false);

        for (int currenTileIdx = 0; currenTileIdx < actualAmountTiles; currenTileIdx++) {
            int bytesToRead = (tileOffsets[currenTileIdx] - byteReader.getTotalBytesRead()) + bytesPerTile[currenTileIdx];
            // Mark the current position in the stream
            // This is used to reset the stream to this position after reading the data
            // Valid until bytesToRead + 1 bytes are read
            byteReader.mark(bytesToRead);
            // Read until offset is reached
            byteReader.readBytes(tileOffsets[currenTileIdx] - byteReader.getTotalBytesRead());
            byte[] currentTileData = byteReader.readBytes(bytesPerTile[currenTileIdx]);

            // Reset the stream to the beginning of the next tile
            byteReader.reset();

            // TODO: If the tile is compressed, decompress the currentTileData here

            if (compression == 8) {
                currentTileData = COGCompressionUtils.decompressDeflate(currentTileData);
            }

            int pixelsRead = 0;
            int bytesRead = 0;
            int currentRow = 0;
            if (planarConfiguration == 1) {
                // Interleaved
                // RGBRGBRGB
                while (currentRow < tileLength && pixelsRead < tileWidth) {
                    for (int bandIdx = 0; bandIdx < bands; bandIdx++) {
                        double value = 0;
                        int sampleLength = bitsPerSample[bandIdx] / 8;

                        switch (sampleFormat[bandIdx]) {
                            case UNSIGNED_INTEGER:
                            case UNDEFINED:
                                // According to the standard, this should be handled as not being there -> 1 (unsigned integer)
                                value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, false, false, false).doubleValue();
                                break;
                            case SIGNED_INTEGER:
                                value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, false, true, false).doubleValue();
                                break;
                            case FLOATING_POINT:
                                value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, true, false, false).doubleValue();
                                break;
                        }

                        bytesRead += sampleLength;
                        outputMatrix.set((currentTileRow * tileLength) + currentRow,
                                (currentTileCol * tileWidth * bands) + (pixelsRead * bands) + bandIdx,
                                value);
                    }

                    pixelsRead++;
                    if (pixelsRead >= tileWidth) {
                        pixelsRead = 0;
                        currentRow++;
                    }
                }
            } else if (planarConfiguration == 2 && calculatedAmountTiles * bands == tileOffsets.length) {
                // If every band is stored in different tiles, so first one R, second one G and so on
                // RRRGGGBBB
                // TODO: Currently this doesn't seem standardized properly, there are still open GitHub issues about that
                // e.g.: https://github.com/cogeotiff/cog-spec/issues/17
                // if something changes in the standard, this may need to be adjusted, interleaved is discouraged in COG though
                if (currenTileIdx - (currentBand * calculatedAmountTiles) >= calculatedAmountTiles) {
                    currentTileCol = 0;
                    currentTileRow = 0;
                    currentBand++;
                }

                int sampleLength = bitsPerSample[currentBand] / 8;

                while (currentRow < tileLength && pixelsRead < tileWidth) {
                    double value = 0;

                    switch (sampleFormat[currentBand]) {
                        case UNSIGNED_INTEGER:
                        case UNDEFINED:
                            // According to the standard, this should be handled as not being there -> 1 (unsigned integer)
                            value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, false, false, false).doubleValue();
                            break;
                        case SIGNED_INTEGER:
                            value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, false, true, false).doubleValue();
                            break;
                        case FLOATING_POINT:
                            value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, true, false, false).doubleValue();
                            break;
                    }
                    bytesRead += sampleLength;
                    outputMatrix.set((currentTileRow * tileLength) + currentRow,
                            (currentTileCol * tileWidth * bands) + (pixelsRead * bands) + currentBand,
                            value);
                    pixelsRead++;
                    if (pixelsRead >= tileWidth) {
                        pixelsRead = 0;
                        currentRow++;
                    }
                }
            } else {
                throw new DMLRuntimeException("Unsupported Planar Configuration: " + planarConfiguration);
            }

            currentTileCol++;
            if (currentTileCol >= tileCols) {
                currentTileCol = 0;
                currentTileRow++;
            }

        }

        outputMatrix.examSparsity();
        return outputMatrix;
    }
}
