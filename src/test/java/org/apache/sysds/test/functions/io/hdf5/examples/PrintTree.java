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


import org.apache.sysds.runtime.io.hdf5.HDF5File;
import org.apache.sysds.runtime.io.hdf5.api.Group;
import org.apache.sysds.runtime.io.hdf5.api.Node;

import java.io.File;
import java.util.Collections;

/**
 * An example of recursively parsing a HDF5 file tree and printing it to the
 * console.
 *
 * @author James Mudd
 */
public class PrintTree {

	public static void main(String[] args) {
		String filePath="/home/sfathollahzadeh/sample/adbc.h5";
		File file = new File(filePath);
		System.out.println(file.getName());

		try (HDF5File HDF5File = new HDF5File(file)) {
			recursivePrintGroup(HDF5File, 0);
		}
	}

	private static void recursivePrintGroup(Group group, int level) {
		level++;
		String indent = String.join("", Collections.nCopies(level, "    "));
		for (Node node : group) {
			System.out.println(indent + node.getName());
			if (node instanceof Group) {
				recursivePrintGroup((Group) node, level);
			}
		}
	}

}
