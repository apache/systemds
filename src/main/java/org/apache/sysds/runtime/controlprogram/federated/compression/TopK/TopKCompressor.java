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

package org.apache.sysds.runtime.controlprogram.federated.compression.TopK;

import org.apache.sysds.runtime.controlprogram.federated.compression.CompressedMatrix;
import org.apache.sysds.runtime.controlprogram.federated.compression.CompressionType;
import org.apache.sysds.runtime.controlprogram.federated.compression.MatrixCompressor;
import org.apache.sysds.runtime.controlprogram.federated.compression.exceptions.CompressionException;
import org.apache.sysds.runtime.controlprogram.federated.compression.exceptions.DecompressionException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * TopK Sparsification Compressor.
 *
 * Keeps only the K largest-magnitude elements in the matrix, setting all others to zero. Optimal for gradient
 * sparsification in federated learning where most gradient values are near zero.
 *
 * Compression ratio: approximately 1/sparsityRatio e.g. sparsityRatio=0.01 keeps 1% of elements → ~100x compression
 */
public class TopKCompressor implements MatrixCompressor {

	private final double sparsityRatio; // Fraction of elements to keep (0, 1]
	private final boolean useHeap; // Use min-heap for O(n log k) selection

	/**
	 * @param sparsityRatio Fraction of elements to retain e.g. 0.01 = keep top 1%
	 * @param useHeap       If true, use priority queue (faster for large matrices)
	 */
	public TopKCompressor(double sparsityRatio, boolean useHeap) {
		if(sparsityRatio <= 0 || sparsityRatio > 1) {
			throw new IllegalArgumentException("sparsityRatio must be in (0, 1]");
		}
		this.sparsityRatio = sparsityRatio;
		this.useHeap = useHeap;
	}

	/** Default constructor: uses heap-based selection */
	public TopKCompressor(double sparsityRatio) {
		this(sparsityRatio, true);
	}

	@Override
	public CompressedMatrix compress(MatrixBlock input) throws CompressionException {
		try {
			int numRows = input.getNumRows();
			int numCols = input.getNumColumns();
			int totalElements = numRows * numCols;
			int k = (int) Math.max(1, Math.ceil(totalElements * sparsityRatio));

			// If k covers everything, no compression needed
			if(k >= totalElements) {
				return new CompressedMatrix(CompressionType.TOPK, numRows, numCols, input, 1.0);
			}

			// Extract all non-zero elements with their linear indices
			List<Element> elements = extractElements(input, numRows, numCols);

			// If fewer non-zeros than k, keep all of them
			List<Element> topK = (elements.size() <= k) ? new ArrayList<>(elements) : selectTopK(elements, k);

			// Pack into TopKData
			TopKData data = convertToTopKData(topK, numCols);

			double ratio = calculateCompressionRatio(totalElements, topK.size());

			return new CompressedMatrix(CompressionType.TOPK, numRows, numCols, data, ratio);

		}
		catch(Exception e) {
			throw new CompressionException("TopK compression failed: " + e.getMessage(), e);
		}
	}

	@Override
	public MatrixBlock decompress(CompressedMatrix compressed) throws DecompressionException {
		try {
			// Handle passthrough case (no compression was applied)
			if(compressed.getCompressedData() instanceof MatrixBlock) {
				return (MatrixBlock) compressed.getCompressedData();
			}

			TopKData data = (TopKData) compressed.getCompressedData();
			MatrixBlock result = new MatrixBlock(compressed.getNumRows(), compressed.getNumCols(), true // Start sparse
			);
			result.allocateSparseRowsBlock();

			// Place values back at their original positions
			for(int i = 0; i < data.indices.length; i++) {
				int linearIdx = data.indices[i];
				int row = linearIdx / data.numCols;
				int col = linearIdx % data.numCols;
				result.set(row, col, data.values[i]);
			}

			result.examSparsity();
			return result;

		}
		catch(ClassCastException e) {
			throw new DecompressionException("Invalid compressed data type for TopK", e);
		}
		catch(Exception e) {
			throw new DecompressionException("TopK decompression failed: " + e.getMessage(), e);
		}
	}

	// -----------------------------------------------------------------------
	// Private helpers
	// -----------------------------------------------------------------------

	/**
	 * Extract all non-zero elements with their linear indices. Handles both dense and sparse MatrixBlock
	 * representations.
	 */
	private List<Element> extractElements(MatrixBlock input, int numRows, int numCols) {
		List<Element> elements = new ArrayList<>();

		if(input.isInSparseFormat()) {
			// Sparse: iterate only over non-zero entries
			for(int i = 0; i < numRows; i++) {
				if(input.getSparseBlock() == null)
					continue;
				if(input.getSparseBlock().isEmpty(i))
					continue;
				int[] rowIndices = input.getSparseBlock().indexes(i);
				double[] rowValues = input.getSparseBlock().values(i);
				int rowSize = input.getSparseBlock().size(i);
				for(int j = 0; j < rowSize; j++) {
					double val = rowValues[j];
					if(val != 0.0) {
						int linearIdx = i * numCols + rowIndices[j];
						elements.add(new Element(linearIdx, val, Math.abs(val)));
					}
				}
			}
		}
		else {
			// Dense: iterate all elements, skip zeros
			double[] denseBlock = input.getDenseBlockValues();
			if(denseBlock != null) {
				for(int i = 0; i < denseBlock.length; i++) {
					if(denseBlock[i] != 0.0) {
						elements.add(new Element(i, denseBlock[i], Math.abs(denseBlock[i])));
					}
				}
			}
		}
		return elements;
	}

	/**
	 * Select top K elements by absolute value. Uses min-heap for O(n log k) when useHeap=true, or full sort O(n log n)
	 * otherwise.
	 */
	private List<Element> selectTopK(List<Element> elements, int k) {
		if(useHeap) {
			PriorityQueue<Element> minHeap = new PriorityQueue<>(k, Comparator.comparingDouble(e -> e.absValue));
			for(Element e : elements) {
				if(minHeap.size() < k) {
					minHeap.offer(e);
				}
				else if(e.absValue > minHeap.peek().absValue) {
					minHeap.poll();
					minHeap.offer(e);
				}
			}
			List<Element> result = new ArrayList<>(minHeap);
			result.sort(Comparator.comparingInt(e -> e.index));
			return result;
		}
		else {
			elements.sort((a, b) -> Double.compare(b.absValue, a.absValue));
			return new ArrayList<>(elements.subList(0, k));
		}
	}

	private TopKData convertToTopKData(List<Element> topK, int numCols) {
		int[] indices = new int[topK.size()];
		double[] values = new double[topK.size()];
		for(int i = 0; i < topK.size(); i++) {
			indices[i] = topK.get(i).index;
			values[i] = topK.get(i).value;
		}
		return new TopKData(indices, values, numCols);
	}

	private double calculateCompressionRatio(int total, int kept) {
		if(kept == 0)
			return Double.MAX_VALUE;
		// Original: total * 8 bytes (doubles)
		// Compressed: kept * 12 bytes (int index + double value)
		long originalBytes = (long) total * 8;
		long compressedBytes = (long) kept * 12;
		return (double) originalBytes / compressedBytes;
	}

	// -----------------------------------------------------------------------
	// Inner class: element tracking during compression
	// -----------------------------------------------------------------------

	private static class Element {
		final int index; // Linear index: row * numCols + col
		final double value; // Original value
		final double absValue; // Absolute value for magnitude comparison

		Element(int index, double value, double absValue) {
			this.index = index;
			this.value = value;
			this.absValue = absValue;
		}
	}
}
