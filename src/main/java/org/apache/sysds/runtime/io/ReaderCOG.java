package org.apache.sysds.runtime.io;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.cog.COGHeader;
import org.apache.sysds.runtime.io.cog.IFDTag;
import org.apache.sysds.runtime.io.cog.IFDTagDictionary;
import org.apache.sysds.runtime.io.cog.TIFFDataTypes;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;

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
        int magic = cogHeader.parseBytes(header, 2, 2);
        if (magic != 42) {
            throw new RuntimeException("Invalid Magic Number");
        }

        // Read offset of the first IFD
        // Usually this is 8 (right after the header) we are at right now
        // With COG, GDAL usually writes some metadata before the IFD
        byte[] ifdOffsetRaw = readBytes(bis, 4);
        int ifdOffset = cogHeader.parseBytes(ifdOffsetRaw, 4);

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
            numberOfTags = cogHeader.parseBytes(numberOfTagsRaw, 2);
            ifdTags = new IFDTag[numberOfTags];

            // Read the tags
            for (int i = 0; i < numberOfTags; i++) {
                // Read the tag fully (12 bytes long)
                // 2 bytes tag ID
                // 2 bytes data type
                // 4 bytes data count
                // 4 bytes data value (can also be offset)
                byte[] tag = readBytes(bis, 12);
                int tagId = cogHeader.parseBytes(tag, 2);

                int tagType = cogHeader.parseBytes(tag, 2, 2);
                TIFFDataTypes dataType = TIFFDataTypes.valueOf(tagType);

                int tagCount = cogHeader.parseBytes(tag, 4, 4);

                int tagValue = cogHeader.parseBytes(tag, 4, 8);
                int[] tagData = new int[tagCount];

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
                        tagData[j] = cogHeader.parseBytes(data, dataType.getSize(), j * dataType.getSize());
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

                // For the bits per sample the actual data can be encoded in the length
                // but for now what also works is the offset, there we then have [8,8,8] e.g.
                // not sure how to handle that perfectly
                //if (tagId == IFDTagDictionary.BitsPerSample.getValue()) {
                // Do something
                //}
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
            nextIFDOffset = cogHeader.parseBytes(nextIFDOffsetRaw, 4);
        }

        // TODO: Actually read image data
        // We can get this from the tile offsets and tile byte counts


        int rows = 10;
        int cols = 10;
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
