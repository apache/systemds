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

public class ReaderCOG extends MatrixReader{
    protected final FileFormatPropertiesCOG _props;
    private int totalBytesRead = 0;

    public ReaderCOG(FileFormatPropertiesCOG props) {
        _props = props;
    }
    @Override
    public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {
        JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
        Path path = new Path(fname);
        FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

        BufferedInputStream bis = new BufferedInputStream(fs.open(path));
        return readCOG(bis, rlen, clen, blen, estnnz);
    }

    @Override
    public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {
        BufferedInputStream bis = new BufferedInputStream(is);
        return readCOG(bis, rlen, clen, blen, estnnz);
    }

    private MatrixBlock readCOG(BufferedInputStream bis, long rlen, long clen, int blen, long estnnz) {
        // Read first 4 bytes to determine byte order and make sure it is a valid TIFF
        byte[] header = readBytes(bis, 4);


        // First read the byte order
        // TODO: Make this nicer, maybe in the COGHeader class as a static method?
        boolean littleEndian = false;
        if ((header[0] & 0xFF) == 0x4D && (header[1] & 0xFF) == 0x4D) {
            // TODO: Remove this
            System.out.println("Big-Endian detected (MM)");
        } else if ((header[0] & 0xFF) == 0x49 && (header[1] & 0xFF) == 0x49) {
            // TODO: Remove this
            System.out.println("Little-Endian detected (II)");
            littleEndian = true;
        } else {
            throw new RuntimeException("Invalid Byte-Order");
        }

        // Create COGHeader object, initialize with the correct byte order
        COGHeader cogHeader = new COGHeader(littleEndian);

        // Check magic number (42), otherwise this is not a valid TIFF
        int magic = cogHeader.parseByteArray(header, 2, 2, false, false, false).intValue();
        if (magic != 42) {
            throw new RuntimeException("Invalid Magic Number");
        }

        // Read offset of the first IFD
        // Usually this is 8 (right after the header) we are at right now
        // With COG, GDAL usually writes some metadata before the IFD
        byte[] ifdOffsetRaw = readBytes(bis, 4);
        int ifdOffset = cogHeader.parseByteArray(ifdOffsetRaw, 4, 0, false, false, false).intValue();

        // If the IFD offset is larger than 8, read that and store it in the COGHeader
        // This is the GDAL metadata
        if (ifdOffset > 8) {
            // Read the metadata from the current position to the IFD offset
            // -8 because the offset is calculated from the beginning of the file
            byte[] metadata = readBytes(bis, ifdOffset - 8);
            cogHeader.setGDALMetadata(new String(metadata));
        }

        // If we read the first IFD, we handle it somewhat differently
        // See the if-statement below
        boolean firstIFD = true;
        // Is used at the end of the while loop to determine if there is another IFD
        byte[] nextIFDOffsetRaw;
        int nextIFDOffset = 0;

        // Used in the beginning of the while loop to read the number of tags in the IFD
        byte[] numberOfTagsRaw;
        int numberOfTags;
        // Array to store the IFD tags, initialized after the number of tags were read
        IFDTag[] ifdTags;

        // Read the IFDs, always read the first one
        // The nextIFDOffset ist 0 if there is no next IFD
        while (nextIFDOffset != 0 || firstIFD) {
            // TODO: Find out what data this is
            // For some reason there is data between the IFDs. And I don't know what data or why.
            // Let's ignore it for now and find out later
            // Read until the next IFD, discard any data until then
            readBytes(bis, nextIFDOffset - (firstIFD ? 0 : totalBytesRead));

            // Read the number of tags in the IFD and initialize the array
            numberOfTagsRaw = readBytes(bis, 2);
            numberOfTags = cogHeader.parseByteArray(numberOfTagsRaw, 2, 0, false, false, false).intValue();
            ifdTags = new IFDTag[numberOfTags];

            // Read the tags
            for (int i = 0; i < numberOfTags; i++) {
                // Read the tag fully (12 bytes long)
                // 2 bytes tag ID
                // 2 bytes data type
                // 4 bytes data count
                // 4 bytes data value (can also be offset)
                byte[] tag = readBytes(bis, 12);
                int tagId = cogHeader.parseByteArray(tag, 2, 0, false, false, false).intValue();

                int tagType = cogHeader.parseByteArray(tag, 2, 2, false, false, false).intValue();
                TIFFDataTypes dataType = TIFFDataTypes.valueOf(tagType);

                int tagCount = cogHeader.parseByteArray(tag, 4, 4, false, false, false).intValue();

                int tagValue = cogHeader.parseByteArray(tag, 4, 8, false, false, false).intValue();
                Number[] tagData = new Number[tagCount];

                // If the data in total is larger than 4 bytes it is an offset to the actual data
                if (dataType.getSize() * tagCount > 4) {
                    // Read the data from the offset
                    // tagValue = offset, just assigning this for better readability
                    int offset = tagValue;
                    // data length = tagCount * data type size
                    int totalSize = tagCount * dataType.getSize();

                    // Calculate the number of bytes to read in order to reset our reader
                    // after going to that offset
                    int bytesToRead = (offset - totalBytesRead) + totalSize;

                    // Mark the current position in the stream
                    // This is used to reset the stream to this position after reading the data
                    // Valid until bytesToRead + 1 bytes are read
                    bis.mark(bytesToRead + 1);
                    // Read until offset is reached
                    readBytes(bis, offset - totalBytesRead);
                    // Read actual data
                    byte[] data = readBytes(bis, totalSize);

                    // Read the data with the given size of the data type
                    for (int j = 0; j < tagCount; j++) {
                        switch (dataType) {
                            // All unsigned non-floating point values
                            case BYTE:
                            case ASCII:
                            case SHORT:
                            case LONG:
                            case UNDEFINED:
                                tagData[j] = cogHeader.parseByteArray(data, dataType.getSize(), j * dataType.getSize(), false, false, false);
                                break;
                            case RATIONAL:
                                tagData[j] = cogHeader.parseByteArray(data, dataType.getSize(), j * dataType.getSize(), false, false, true);
                                break;
                            case SBYTE:
                            case SSHORT:
                            case SLONG:
                                tagData[j] = cogHeader.parseByteArray(data, dataType.getSize(), j * dataType.getSize(), false, true, false);
                                break;
                            case SRATIONAL:
                                tagData[j] = cogHeader.parseByteArray(data, dataType.getSize(), j * dataType.getSize(), false, true, true);
                                break;
                            case FLOAT:
                            case DOUBLE:
                                tagData[j] = cogHeader.parseByteArray(data, dataType.getSize(), j * dataType.getSize(), true, false, false);
                                break;
                        }
                    }

                    // Reset the stream to the beginning of the next tag
                    try {
                        bis.reset();
                        totalBytesRead -= bytesToRead;
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                } else { // If the data fits in the 4 bytes
                    tagData[0] = tagValue;
                }
                // Read the tag ID and get the corresponding tag from the dictionary (enum)
                IFDTagDictionary tagDictionary = IFDTagDictionary.valueOf(tagId);

                // Currently we'll just throw away any tag that doesn't fit here
                if (tagDictionary == null) {
                    String doSomething = "?????";
                }

                // Create the constructed IFDTag object and store it in the array
                IFDTag ifdTag = new IFDTag(tagDictionary, (short) tagType, tagCount, tagData);
                ifdTags[i] = ifdTag;
            }
            if (firstIFD) {
                // If this is the first IFD, set it as the main IFD in the COGHeader
                cogHeader.setIFD(ifdTags.clone());
                firstIFD = false;
            } else {
                // If this is not the first IFD, add it as an additional IFD
                cogHeader.addAdditionalIFD(ifdTags.clone());
            }
            // Read the offset to the next IFD. If it is 0, there is no next IFD
            nextIFDOffsetRaw = readBytes(bis, 4);
            nextIFDOffset = cogHeader.parseByteArray(nextIFDOffsetRaw, 4, 0, false, false, false).intValue();
        }

        // Check compatibility of the file with our reader
        // Certain options are not supported, and we need to filter out some non-standard options
        String isCompatible = COGHeader.isCompatible(cogHeader.getIFD());
        if (!isCompatible.equals("")) {
            throw new RuntimeException("Incompatible COG file: " + isCompatible);
        }

        // TODO: Actually read image data
        // We can get this from the tile offsets and tile byte counts

        int rows = -1;
        int cols = -1;
        int bands = -1;
        int[] bitsPerSample = null;
        SampleFormatDataTypes[] sampleFormat = null;
        int tileWidth = -1;
        int tileLength = -1;
        int[] tileOffsets = null;
        int[] tileByteCounts = null;

        for (IFDTag ifd : cogHeader.getIFD()) {
            IFDTagDictionary tag = ifd.getTagId();
            if (tag == IFDTagDictionary.ImageWidth) {
                cols = ifd.getData()[0].intValue();
            }
            else if(tag == IFDTagDictionary.ImageLength) {
                rows = ifd.getData()[0].intValue();
            }
            // = Number of bands effectively
            else if(tag == IFDTagDictionary.SamplesPerPixel){
                bands = ifd.getData()[0].intValue();
            }
            else if(tag == IFDTagDictionary.BitsPerSample) {
                bitsPerSample = Arrays.stream(ifd.getData()).mapToInt(Number::intValue).toArray();
            }
            else if(tag == IFDTagDictionary.TileWidth) {
                tileWidth = ifd.getData()[0].intValue();
            }
            else if(tag == IFDTagDictionary.TileLength) {
                tileLength = ifd.getData()[0].intValue();
            }
            else if(tag == IFDTagDictionary.TileOffsets) {
                tileOffsets = Arrays.stream(ifd.getData()).mapToInt(Number::intValue).toArray();
            }
            else if(tag == IFDTagDictionary.TileByteCounts) {
                tileByteCounts = Arrays.stream(ifd.getData()).mapToInt(Number::intValue).toArray();
            } else if(tag == IFDTagDictionary.SampleFormat) {
                int dataCount = ifd.getDataCount();
                sampleFormat = new SampleFormatDataTypes[dataCount];
                for (int i = 0; i < dataCount; i++) {
                    sampleFormat[i] = SampleFormatDataTypes.valueOf(ifd.getData()[i].intValue());
                }
            }
        }

        MatrixBlock[] dmatrix = new MatrixBlock[bands];

        for (int i = 0; i < bands; i++) {
            dmatrix[i] = new MatrixBlock(rows, cols, false);
        }

        int bytesToRead = (tileOffsets[0] - totalBytesRead) + tileByteCounts[0];

        // Mark the current position in the stream
        // This is used to reset the stream to this position after reading the data
        // Valid until bytesToRead + 1 bytes are read
        bis.mark(bytesToRead + 1);
        // Read until offset is reached
        readBytes(bis, tileOffsets[0] - totalBytesRead);
        byte[] firstTileData = readBytes(bis, tileByteCounts[0]);

        // Reset the stream to the beginning of the next tag
        try {
            bis.reset();
            totalBytesRead -= bytesToRead;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int pixelsRead = 0;
        int currentRow = 0;
        int bytesRead = 0;
        while (bytesRead < firstTileData.length){
            for (int i = 0; i < bands; i++) {
                double value = 0;
                int sampleLength = bitsPerSample[i] / 8;
                switch (sampleFormat[i]) {
                    case UNSIGNED_INTEGER:
                    case UNDEFINED:
                        // According to the standard, this should be handled as not being there -> 1 (unsigned integer)
                        value = cogHeader.parseByteArray(firstTileData, sampleLength, bytesRead, false, false, false).doubleValue();
                        break;
                    case SIGNED_INTEGER:
                        value = cogHeader.parseByteArray(firstTileData, sampleLength, bytesRead, false, true, false).doubleValue();
                        break;
                    case FLOATING_POINT:
                        value = cogHeader.parseByteArray(firstTileData, sampleLength, bytesRead, true, false, false).doubleValue();
                        break;
                }
                bytesRead += sampleLength;
                dmatrix[i].set(currentRow, pixelsRead, value);
                // pixelsRead, currentRow

            }

            pixelsRead++;
            if (pixelsRead >= tileWidth) {
                pixelsRead = 0;
                currentRow++;
            }
        }


        MatrixBlock dummyMatrix = new MatrixBlock(rows, cols, false);
        dummyMatrix.allocateDenseBlock();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dummyMatrix.set(i, j, 1.0);
            }
        }
        // Isn't printed when debugging (or running) the tests
        // I'll leave it for now
        System.out.println("We done something!");
        return dummyMatrix;
    }

    /**
     * Reads a given number of bytes from the BufferedInputStream.
     * Increments the totalBytesRead counter by the number of bytes read.
     * @param bis
     * @param length
     * @return
     */
    private byte[] readBytes(BufferedInputStream bis, int length) {
        byte[] header = new byte[length];
        try {
            bis.read(header);
            totalBytesRead += length;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return header;
    }
}
