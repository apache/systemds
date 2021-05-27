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
import org.apache.sysds.runtime.io.hdf5.api.Attribute;
import org.apache.sysds.runtime.io.hdf5.api.Node;
import org.apache.commons.lang3.ArrayUtils;

import java.io.File;

/**
 * Example application for reading an attribute from HDF5
 *
 * @author James Mudd
 */
public class ReadAttribute {

	/**
	 * @param args ["path/to/file.hdf5", "path/to/node", "attributeName"]
	 */
	public static void main(String[] args) {
		File file = new File(args[0]);

		try (HDF5File HDF5File = new HDF5File(file)) {
			Node node = HDF5File.getByPath(args[1]);
			Attribute attribute = node.getAttribute(args[2]);
			Object attributeData = attribute.getData();
			System.out.println(ArrayUtils.toString(attributeData));
		}
	}
}
