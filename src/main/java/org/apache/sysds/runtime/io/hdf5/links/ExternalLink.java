/*
 * This file is part of jHDF. A pure Java library for accessing HDF5 files.
 *
 * http://jhdf.io
 *
 * Copyright (c) 2020 James Mudd
 *
 * MIT License see 'LICENSE' file
 */
package org.apache.sysds.runtime.io.hdf5.links;

import org.apache.sysds.runtime.io.hdf5.HDF5File;
import org.apache.sysds.runtime.io.hdf5.api.Group;
import org.apache.sysds.runtime.io.hdf5.api.Node;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfBrokenLinkException;
import org.apache.commons.lang3.concurrent.LazyInitializer;

import java.io.File;
import java.nio.file.Paths;

/**
 * Link to a {@link Node} in an external HDF5 file. The link is made of both a
 * target HDF5 file and a target path to a {@link Node} within the target file.
 *
 * @author James Mudd
 */
public class ExternalLink extends AbstractLink {

	private final String targetFile;
	private final String targetPath;

	public ExternalLink(String targetFile, String targetPath, String name, Group parent) {
		super(name, parent);
		this.targetFile = targetFile;
		this.targetPath = targetPath;

		targetNode = new ExternalLinkTargetLazyInitializer();
	}

	private class ExternalLinkTargetLazyInitializer extends LazyInitializer<Node> {
		@Override
		protected Node initialize()  {
			// Open the external file
			final HDF5File externalFile = new HDF5File(getTargetFile());
			// Tell this file about it to keep track of open external files
			getHdfFile().addExternalFile(externalFile);
			return externalFile.getByPath(targetPath);
		}

		private File getTargetFile() {
			// Check if the target file path is absolute
			if (targetFile.startsWith(File.separator)) {
				return Paths.get(targetFile).toFile();
			} else {
				// Need to resolve the full path
				String absolutePathOfThisFilesDirectory = parent.getFile().getParent();
				return Paths.get(absolutePathOfThisFilesDirectory, targetFile).toFile();
			}
		}
	}

	@Override
	public Node getTarget() {
		try {
			return targetNode.get();
		} catch (Exception e) {
			throw new HdfBrokenLinkException(
					"Could not resolve link target '" + targetPath + "' in external file '" + targetFile
							+ "' from link '" + getPath() + "'",
					e);
		}
	}

	@Override
	public String getTargetPath() {
		return targetFile + ":" + targetPath;
	}

	@Override
	public String toString() {
		return "ExternalLink [name=" + name + ", targetFile=" + targetFile + ", targetPath=" + targetPath + "]";
	}

}
