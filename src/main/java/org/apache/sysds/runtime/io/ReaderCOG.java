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

        // Check magic number (42)
        int magic = cogHeader.parseBytes(header, 2, 2);
        if (magic != 42) {
            throw new RuntimeException("Invalid Magic Number");
        }
        // Read IFD Offset
        // Usually this is 8 (right after the header) we are at right now
        // With COG, GDAL usually writes some metadata before the IFD
        byte[] ifdOffsetRaw = readBytes(bis, 4);
        int ifdOffset = cogHeader.parseBytes(ifdOffsetRaw, 4);

        // If the IFD offset is larger than 8, read that and store it in the COGHeader
        if (ifdOffset > 8) {
            // Read the metadata from the current position to the IFD offset
            // -8 because the offset is calculated from the beginning of the file
            byte[] metadata = readBytes(bis, ifdOffset - 8);
            cogHeader.setGDALMetadata(new String(metadata));
        }
        boolean firstIFD = true;
        byte[] nextIFDOffsetRaw;
        int nextIFDOffset = 0;
        byte[] numberOfTagsRaw;
        int numberOfTags;
        IFDTag[] ifdTags;

        // If there is another IFD, read it
        // The nextIFDOffset ist 0 if there is no next IFD
        while (nextIFDOffset != 0 || firstIFD) {
            // TODO: Find out what data this is
            // For some reason there is data between the IFDs. And I don't know what data or why.
            // Let's ignore it for now and find out later
            readBytes(bis, nextIFDOffset - (firstIFD ? 0 : totalBytesRead));

            // Read the next IFD
            numberOfTagsRaw = readBytes(bis, 2);
            numberOfTags = cogHeader.parseBytes(numberOfTagsRaw, 2);
            ifdTags = new IFDTag[numberOfTags];
            for (int i = 0; i < numberOfTags; i++) {
                byte[] tag = readBytes(bis, 12);
                int tagId = cogHeader.parseBytes(tag, 2);

                int tagType = cogHeader.parseBytes(tag, 2, 2);
                TIFFDataTypes dataType = TIFFDataTypes.valueOf(tagType);

                int tagCount = cogHeader.parseBytes(tag, 4, 4);

                // TODO: Implement that this can also be an offset to the data
                int tagValue = cogHeader.parseBytes(tag, 4, 8);
                int[] tagData = new int[tagCount];

                // TODO: Implement multiple tagCount, currently only count = 1 is supported
                // If the tagCount * the data type size is larger than 4, the data is stored in the offset
                if (dataType.getSize() * tagCount > 4) {
                    // Read the data from the offset
                    // tagValue = offset
                    int offset = tagValue;
                    // data length = tagCount * data type size
                    int totalSize = tagCount * dataType.getSize();

                    // Calculate the number of bytes to read in order to reset our reader
                    // after going to that offset
                    int bytesToRead = (offset - totalBytesRead) + totalSize;
                    bis.mark(bytesToRead + 1);
                    // Read until offset is reached
                    readBytes(bis, offset - totalBytesRead);
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
                } else {
                    tagData[0] = tagValue;
                }
                IFDTagDictionary tagDictionary = IFDTagDictionary.valueOf(tagId);
                // For now: throw an exception if the tag is unknown
                // TODO: Maybe good to fail silently here?
                // Currently we'll just throw away any tag that doesn't fit here
                if (tagDictionary == null) {
                    String doSomethng = "";
                    //throw new RuntimeException("Unknown Tag ID: " + tagId);
                }
                // For the bits per sample the actual data is encoded in the length
                //if (tagId == IFDTagDictionary.BitsPerSample.getValue()) {
                // Do something
                //}
                IFDTag ifdTag = new IFDTag(tagDictionary, (short) tagType, tagCount, tagData);
                ifdTags[i] = ifdTag;
            }
            if (firstIFD) {
                cogHeader.setIFD(ifdTags.clone());
                firstIFD = false;
            } else {
                cogHeader.addAdditionalIFD(ifdTags.clone());
            }
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
