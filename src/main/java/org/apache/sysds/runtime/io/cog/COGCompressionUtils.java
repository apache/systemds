package org.apache.sysds.runtime.io.cog;

import org.apache.sysds.runtime.DMLRuntimeException;

import java.io.ByteArrayOutputStream;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

public class COGCompressionUtils {
    /**
     * Decompresses a byte array that was compressed using the Deflate algorithm
     * @param compressedData
     * @return
     * @throws DMLRuntimeException
     */
    public static byte[] decompressDeflate(byte[] compressedData) throws DMLRuntimeException {
        Inflater inflater = new Inflater();
        inflater.setInput(compressedData);

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream(compressedData.length);
        byte[] buffer = new byte[1024];

        while (!inflater.finished()) {
            int decompressedSize = 0;
            try {
                decompressedSize = inflater.inflate(buffer);
            } catch (DataFormatException e) {
                throw new DMLRuntimeException("Failed to decompress tile data", e);
            }
            outputStream.write(buffer, 0, decompressedSize);
        }

        inflater.end();

        return outputStream.toByteArray();
    }
}
