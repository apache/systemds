package org.apache.sysds.runtime.io.cog;

import org.apache.sysds.runtime.DMLRuntimeException;

import java.io.BufferedInputStream;
import java.io.IOException;

public class COGByteReader {
    private int totalBytesRead;
    private BufferedInputStream bis;
    private int readlimit = 0;

    public COGByteReader(BufferedInputStream bis) {
        this.bis = bis;
        totalBytesRead = 0;
    }

    public COGByteReader(BufferedInputStream bis, int totalBytesRead) {
        this.bis = bis;
        this.totalBytesRead = totalBytesRead;
    }

    public int getTotalBytesRead() {
        return totalBytesRead;
    }

    public void setTotalBytesRead(int totalBytesRead) {
        this.totalBytesRead = totalBytesRead;
    }

    /**
     * Reads a given number of bytes from the BufferedInputStream.
     * Increments the totalBytesRead counter by the number of bytes read.
     * @param length
     * @return
     */
    public byte[] readBytes(int length) {
        byte[] header = new byte[length];
        try {
            bis.read(header);
            totalBytesRead += length;
        } catch (IOException e) {
            throw new DMLRuntimeException(e);
        }
        return header;
    }

    public byte[] readBytes(long length) {
        if (length > Integer.MAX_VALUE) {
            throw new DMLRuntimeException("Cannot read more than Integer.MAX_VALUE bytes at once");
        }
        return readBytes((int) length);
    }

    public void mark(int readlimit) {
        this.readlimit = readlimit;
        bis.mark(readlimit + 1);
    }

    public void reset() throws DMLRuntimeException {
        try {
            bis.reset();
            totalBytesRead -= this.readlimit;
        } catch (IOException e) {
            throw new DMLRuntimeException(e);
        }
    }
}
