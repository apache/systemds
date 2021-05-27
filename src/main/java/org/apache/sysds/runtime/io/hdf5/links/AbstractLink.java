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
import org.apache.sysds.runtime.io.hdf5.api.*;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.commons.lang3.concurrent.LazyInitializer;

import java.io.File;
import java.util.Map;

/**
 * Base class for {@link Link} implementations.
 *
 * @author James Mudd
 */
public abstract class AbstractLink implements Link {

	protected final String name;
	protected final Group parent;
	protected LazyInitializer<Node> targetNode;

	public AbstractLink(String name, Group parent) {
		this.name = name;
		this.parent = parent;
	}

	@Override
	public Group getParent() {
		return parent;
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public String getPath() {
		return parent.getPath() + name;
	}

	@Override
	public Map<String, Attribute> getAttributes() {
		return getTarget().getAttributes();
	}

	@Override
	public Attribute getAttribute(String name) {
		return getTarget().getAttribute(name);
	}

	@Override
	public NodeType getType() {
		return getTarget().getType();
	}

	@Override
	public boolean isGroup() {
		return getTarget().isGroup();
	}

	@Override
	public File getFile() {
		return parent.getFile();
	}

	@Override
	public HDF5File getHdfFile() {
		return parent.getHdfFile();
	}

	@Override
	public boolean isLink() {
		return true;
	}

	@Override
	public long getAddress() {
		return getTarget().getAddress();
	}

	@Override
	public boolean isBrokenLink() {
		try {
			getTarget(); // Attempt to resolve the link
			return false; // Didn't throw so link isn't broken
		} catch (HdfException e) {
			return true; // Resolving the link failed so its broken
		}
	}

	@Override
	public boolean isAttributeCreationOrderTracked() {
		return getTarget().isAttributeCreationOrderTracked();
	}

}
