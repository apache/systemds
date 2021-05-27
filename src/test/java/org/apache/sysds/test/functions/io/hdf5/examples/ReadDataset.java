/*
 * This file is part of jHDF. A pure Java library for accessing HDF5 files.
 *
 * http://jhdf.io
 *
 * Copyright (c) 2020 James Mudd
 *
 * MIT License see 'LICENSE' file
 */
package org.apache.sysds.test.functions.io.hdf5.examples;

import org.apache.sysds.runtime.io.hdf5.H5;
import org.apache.sysds.runtime.io.hdf5.HDF5File;
import org.apache.sysds.runtime.io.hdf5.api.Dataset;

import java.io.File;

public class ReadDataset {
	public static void main(String[] args) {

		String saeedFilePath="/home/sfathollahzadeh/sample/saeed.h5";
		H5.H5Fcreate(saeedFilePath, (byte) 0, 2,10,10,10,10, "DS1");
		//----------------------------
		//HDF5File hdf5File=new HDF5File();
		//hdf5File.writeHDF();

		String filePath="/home/sfathollahzadeh/sample/H5Ex_D_Hyperslab.h5";
//		//String filePath="/home/sfathollahzadeh/sample/testwrite.h5";
//
		File file = new File(filePath);


		try (HDF5File HDF5File = new HDF5File(file)) {
			Dataset dataset = HDF5File.getDatasetByPath("DS1");
			// data will be a java array of the dimensions of the HDF5 dataset
//			Object data = dataset.getData();
//			System.out.println(ArrayUtils.toString(data));
		}
	}
}
