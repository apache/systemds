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

package org.apache.sysds.runtime.compress.cost;

import java.io.Serializable;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.runtime.compress.workload.Op;
import org.apache.sysds.runtime.compress.workload.OpMetadata;
import org.apache.sysds.runtime.compress.workload.OpSided;
import org.apache.sysds.runtime.compress.workload.WTreeNode;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;

public final class CostEstimatorBuilder implements Serializable {

	private static final long serialVersionUID = 14L;

	protected static final Log LOG = LogFactory.getLog(CostEstimatorBuilder.class.getName());

	protected final InstructionTypeCounter counter;

	public CostEstimatorBuilder(WTreeRoot root) {
		counter = new InstructionTypeCounter();
		if(root.isDecompressing())
			counter.decompressions++;
		for(Op o : root.getOps())
			addOp(1, o, counter);
		for(WTreeNode n : root.getChildNodes())
			addNode(1, n, counter);
	}

	protected ICostEstimate create(int nRows, int nCols, double sparsity, boolean isInSpark) {
		if(isInSpark)
			return new HybridCostEstimator(nRows, nCols, sparsity, counter);
		else
			return new ComputationCostEstimator(nRows, nCols, sparsity, counter);
	}

	public InstructionTypeCounter getCounter() {
		return counter;
	}

	private static void addNode(int count, WTreeNode n, InstructionTypeCounter counter) {
		int mult = n.getReps();
		for(Op o : n.getOps())
			addOp(count * mult, o, counter);
		for(WTreeNode nc : n.getChildNodes())
			addNode(count * mult, nc, counter);
	}

	private static void addOp(int count, Op o, InstructionTypeCounter counter) {
		if(o.isDecompressing()) {
			if(o.isOverlapping())
				counter.overlappingDecompressions += count;
			else
				counter.decompressions += count;
		}
		if(o.isDensifying()) {
			counter.isDensifying = true;
		}

		if(o instanceof OpSided) {
			OpSided os = (OpSided) o;
			if(os.isLeftMM())
				counter.leftMultiplications += count;
			else if(os.isRightMM())
				counter.rightMultiplications += count;
			else
				counter.compressedMultiplications += count;
		}
		else if(o instanceof OpMetadata) {
			// ignore it
		}
		else {
			Hop h = o.getHop();
			if(h instanceof AggUnaryOp) {
				AggUnaryOp agop = (AggUnaryOp) o.getHop();

				switch(agop.getDirection()) {
					case Row:
						counter.scans += count;
						break;
					default:
						counter.dictionaryOps += count;
				}
			}
			else if(h instanceof IndexingOp) {
				IndexingOp idxO = (IndexingOp) h;
				if(idxO.isRowLowerEqualsUpper() && idxO.isColLowerEqualsUpper())
					counter.indexing++;
				else if(idxO.isAllRows())
					counter.dictionaryOps += count; // Technically not correct but better than decompression
			}
			else
				counter.dictionaryOps += count;
		}
	}

	public boolean shouldTryToCompress() {
		int numberOps = 0;
		numberOps += counter.scans + counter.leftMultiplications * 2 + counter.rightMultiplications * 2 +
			counter.compressedMultiplications * 4 + counter.dictionaryOps;
		numberOps -= counter.decompressions + counter.overlappingDecompressions * 2;

		final int nrMultiplications = counter.leftMultiplications + counter.rightMultiplications +
			counter.compressedMultiplications;
		final int nrDecompressions = counter.decompressions + counter.overlappingDecompressions * 2;
		if(counter.decompressions == 0 && counter.rightMultiplications == counter.overlappingDecompressions &&
			numberOps > 10)
			return true;
		if(nrDecompressions > nrMultiplications || (nrDecompressions > 1 && nrMultiplications < 1))
			// This condition is added for l2svm and mLogReg y dataset, that is compressing while it should not.
			return false;
		return numberOps > 4;

	}

	public boolean shouldUseOverlap() {
		final int decompressionsOverall = counter.overlappingDecompressions + counter.decompressions;
		return decompressionsOverall == 0 ||
			decompressionsOverall * 10 < counter.leftMultiplications * 9 + counter.dictionaryOps;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(counter);
		return sb.toString();
	}
}
