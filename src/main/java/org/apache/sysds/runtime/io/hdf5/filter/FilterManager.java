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

import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfFilterException;
import org.apache.sysds.runtime.io.hdf5.object.message.FilterPipelineMessage;
import org.apache.sysds.runtime.io.hdf5.object.message.FilterPipelineMessage.FilterInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * This is a singleton for managing the loaded HDF5 filters.
 *
 * @author James Mudd
 */
public enum FilterManager {; // Enum singleton pattern

	private static final Logger logger = LoggerFactory.getLogger(FilterManager.class);

	private static final Map<Integer, Filter> ID_TO_FILTER = new HashMap<>();

	static {
		logger.info("Initializing HDF5 filters...");

		// Load the built in filters
		addFilter(new DeflatePipelineFilter());
		addFilter(new ByteShuffleFilter());
		addFilter(new FletcherChecksumFilter());
		addFilter(new LzfFilter());

		// Add dynamically loaded filters
		ServiceLoader<Filter> serviceLoader = ServiceLoader.load(Filter.class);
		for (Filter pipelineFilter : serviceLoader) {
			addFilter(pipelineFilter);
		}

		logger.info("Initialized HDF5 filters");
	}

	/**
	 * Adds a filter. This can be used to add dynamically loaded filters. Validates
	 * the passed in filter to ensure in meets the specification, see
	 * {@link Filter}.
	 *
	 * @param filter the filter class to add
	 * @throws HdfFilterException if the filter is not valid
	 */
	public static void addFilter(Filter filter) {
		// Add the filter
		ID_TO_FILTER.put(filter.getId(), filter);

		logger.info("Added HDF5 filter '{}' with ID '{}'", filter.getName(), filter.getId());
	}

	/**
	 * Builds a new pipeline for decoding chunks from a
	 * {@link FilterPipelineMessage}.
	 *
	 * @param filterPipelineMessage message containing the datasets filter
	 *                              specification.
	 * @return the new pipeline
	 * @throws HdfFilterException if a required filter is not available
	 */
	public static FilterPipeline getPipeline(FilterPipelineMessage filterPipelineMessage) {
		List<FilterInfo> filters = filterPipelineMessage.getFilters();

		// Check all the required filters are available
		if(!filters.stream().allMatch(filter -> ID_TO_FILTER.containsKey(filter.getId()))) {
			// Figure out the missing filter
			FilterInfo missingFilterInfo = filters.stream()
                    .filter(filter -> !ID_TO_FILTER.containsKey(filter.getId()))
                    .findFirst() // There should be at least one, that's why were here
                    .orElseThrow(() -> new HdfException("Failed to determine missing filter"));
			throw new HdfFilterException("A required filter is not available: name='" + missingFilterInfo.getName()
                    + "' id=" + missingFilterInfo.getId());
		}

		// Decoding so reverse order
		Collections.reverse(filters);

		// Make the new pipeline
		FilterPipeline pipeline = new FilterPipeline();
		// Add each filter
		filters.forEach(filter -> pipeline.addFilter(ID_TO_FILTER.get(filter.getId()), filter.getData()));

		return pipeline;
	}

}
