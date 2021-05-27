/*
 * This file is part of jHDF. A pure Java library for accessing HDF5 files.
 *
 * http://jhdf.io
 *
 * Copyright (c) 2020 James Mudd
 *
 * MIT License see 'LICENSE' file
 */
package org.apache.sysds.runtime.io.hdf5.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is a placeholder for the checksum filter. Currently a no-op allows datasets with a checksum to be read but the
 * checksum is not validated.
 *
 * @author James Mudd
 */
public class FletcherChecksumFilter implements Filter {
    private static final Logger logger = LoggerFactory.getLogger(FletcherChecksumFilter.class);

    private boolean warningIssued = false;

    @Override
    public int getId() {
        return 3;
    }

    @Override
    public String getName() {
        return "fletcher32";
    }

    @Override
    public byte[] decode(byte[] encodedData, int[] filterData) {
        if(!warningIssued) {
            logger.warn("Fletcher 32 checksum will not be verified");
            warningIssued = true;
        }

        return encodedData;
    }
}
