/*
 * This file is part of jHDF. A pure Java library for accessing HDF5 files.
 *
 * http://jhdf.io
 *
 * Copyright (c) 2020 James Mudd
 *
 * MIT License see 'LICENSE' file
 */
package org.apache.sysds.test.functions.io.hdf5;

import org.apache.sysds.runtime.io.hdf5.*;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;

public class ReadDataset2 {
	public static void main(String[] args) {

//		String hexString = "41630D54FFF68872";
//		//long longBits = 1//Long.valueOf(hexString,16).longValue();
//		double d=1.1;
//		long lb=Long.valueOf(1);
//		double doubleValue = Double.longBitsToDouble(lb);
//		System.out.println(lb);
//		System.out.println( "double float hexString is = " + doubleValue );
//
//		long a= Double.doubleToLongBits(1.1);
//		System.out.println(a);

				write();
		//		test();
		//-----------------------------
//		String datapath = "/home/sfathollahzadeh/sample/H5Ex_D_Hyperslab.h5";
//		H5RootObject rootObject = H5.H5Fopen(datapath);
//		H5ContiguousDataset contiguousDataset = H5.H5Dopen(rootObject, "A1234567890123456");
//		double[][] data = H5.H5Dread(rootObject, contiguousDataset);
//
//		for(int i = 0; i < rootObject.getRow(); i++) {
//			for(int j = 0; j < rootObject.getCol(); j++) {
//				System.out.print(data[i][j] + "  ");
//			}
//			System.out.println();
//		}
//
//		H5.H5Fclose(rootObject);

	}

	private static void test() {
		String orig = "/home/sfathollahzadeh/sample/H5Ex_D_Hyperslab.h5";
		String newData = "/home/sfathollahzadeh/sample/st.h5";
		String ff = "/home/sfathollahzadeh/sample/ff.h5";

		try {
			File origFile = new File(orig);
			FileChannel fcOrig = FileChannel.open(origFile.toPath(), StandardOpenOption.READ);

			File newFile = new File(newData);
			FileChannel fcNew = FileChannel.open(newFile.toPath(), StandardOpenOption.READ);

			ByteBuffer origBB = ByteBuffer.allocate(32);
			ByteBuffer newBB = ByteBuffer.allocate(32);

			fcOrig.read(origBB, 2048);
			fcNew.read(newBB, 2048);

			byte[] origByte = origBB.array();
			byte[] newByte = newBB.array();


			ArrayList<Integer> difs = new ArrayList<>();
			for(int i = 0; i < 32; i++) {
				if(origByte[i] != newByte[i]) {
					difs.add(i);
				}
			}

			int l = difs.size();
			System.out.println("L=" + l);
			for(Integer i : difs) {
				System.out.print(i + ", ");
			}

		}
		catch(Exception exception) {
			exception.printStackTrace();
		}

	}

	private static void write() {
		String datapath = "/home/sfathollahzadeh/sample/st.h5";

		int row = 2;
		int col = 2;
		H5RootObject rootObject = H5.H5Fcreate(datapath);
		H5.H5Screate(rootObject, row, col);
		H5.H5Dcreate(rootObject, row, col, "A1234567890123456");

		double d = 2;
		double[][] data = new double[row][col];
		for(int i = 0; i < row; i++)
			for(int j = 0; j < col; j++)
				data[i][j] = d++;

		H5.H5Dwrite(rootObject, data);
		H5.H5Fclose(rootObject);

	}
}
