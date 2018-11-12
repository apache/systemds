/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.sysml.runtime.instructions.gpu.context;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import jcuda.Pointer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;

public class GPUMatrixMemoryManager {
	protected static final Log LOG = LogFactory.getLog(GPUMatrixMemoryManager.class.getName());
	GPUMemoryManager gpuManager;
	public GPUMatrixMemoryManager(GPUMemoryManager gpuManager) {
		this.gpuManager = gpuManager;
	}
	
	/**
	 * Adds the GPU object to the memory manager
	 * 
	 * @param gpuObj the handle to the GPU object
	 */
	void addGPUObject(GPUObject gpuObj) {
		gpuObjects.add(gpuObj);
	}

	
	/**
	 * list of allocated {@link GPUObject} instances allocated on {@link GPUContext#deviceNum} GPU
	 * These are matrices allocated on the GPU on which rmvar hasn't been called yet.
	 * If a {@link GPUObject} has more than one lock on it, it cannot be freed
	 * If it has zero locks on it, it can be freed, but it is preferrable to keep it around
	 * so that an extraneous host to dev transfer can be avoided
	 */
	HashSet<GPUObject> gpuObjects = new HashSet<>();

	/**
	 * Return a set of GPU Objects associated with a list of pointers
	 * @param pointers A list of pointers
	 * @return A set of GPU objects corresponding to any of these pointers
	 */
	Set<GPUObject> getGpuObjects(Set<Pointer> pointers) {
		Set<GPUObject> gObjs = new HashSet<>();
		for (GPUObject g : gpuObjects) {
			if (!Collections.disjoint(g.getPointers(), pointers))
				gObjs.add(g);
		}
		return gObjs;
	}
	
	Set<GPUObject> getGpuObjects() {
		return gpuObjects;
	}
	
	/**
	 * Return all pointers in the first section
	 * @return all pointers in this section
	 */
	Set<Pointer> getPointers() {
		return gpuObjects.stream().flatMap(gObj -> gObj.getPointers().stream()).collect(Collectors.toSet());
	}
	
	/**
	 * Get pointers from the first memory sections "Matrix Memory"
	 * @param locked return locked pointers if true
	 * @param dirty return dirty pointers if true
	 * @param isCleanupEnabled return pointers marked for cleanup if true
	 * @return set of pointers
	 */
	Set<Pointer> getPointers(boolean locked, boolean dirty, boolean isCleanupEnabled) {
		return gpuObjects.stream().filter(
				gObj -> (gObj.isLocked() == locked && gObj.isDirty() == dirty) ||
						(gObj.mat.isCleanupEnabled() == isCleanupEnabled)).flatMap(
						gObj -> gObj.getPointers().stream()).collect(Collectors.toSet());
	}
	
	/**
	 * Clear all unlocked gpu objects
	 * 
	 * @param opcode instruction code
	 * @throws DMLRuntimeException if error
	 */
	void clearAllUnlocked(String opcode) throws DMLRuntimeException {
		Set<GPUObject> unlockedGPUObjects = gpuObjects.stream()
				.filter(gpuObj -> !gpuObj.isLocked()).collect(Collectors.toSet());
		if(unlockedGPUObjects.size() > 0) {
			if(LOG.isWarnEnabled())
				LOG.warn("Clearing all unlocked matrices (count=" + unlockedGPUObjects.size() + ").");
			for(GPUObject toBeRemoved : unlockedGPUObjects) {
				if(toBeRemoved.dirty)
					toBeRemoved.copyFromDeviceToHost(opcode, true, true);
				else
					toBeRemoved.clearData(opcode, true);
			}
			gpuObjects.removeAll(unlockedGPUObjects);
		}
	}
}
