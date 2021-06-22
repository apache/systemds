package org.apache.sysds.runtime.compress.workload;

import java.util.List;
import java.util.Set;

import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewrite.RewriteCompressedReblock;

public class OpFactory {
	public static Op create(Hop hop, Set<Long> compressed, Set<String> transientCompressed, Set<Long> transposed) {
		if(hop instanceof AggBinaryOp) {
			AggBinaryOp agbhop = (AggBinaryOp) hop;
			List<Hop> in = agbhop.getInput();
			boolean transposedLeft = transposed.contains(in.get(0).getHopID());
			boolean transposedRight = transposed.contains(in.get(1).getHopID());
			boolean left = compressed.contains(in.get(0).getHopID()) ||
				transientCompressed.contains(in.get(0).getName());
			boolean right = compressed.contains(in.get(1).getHopID()) ||
				transientCompressed.contains(in.get(1).getName());
			return new OpSided(hop, left, right, transposedLeft, transposedRight);
		}
		else if(hop.getDataType().isMatrix()) {
			if(HopRewriteUtils.isBinaryMatrixScalarOperation(hop) ||
				HopRewriteUtils.isBinaryMatrixRowVectorOperation(hop))
				return new OpNormal(hop, true);
			else if(hop instanceof IndexingOp) {
				IndexingOp idx = (IndexingOp) hop;
				if(HopRewriteUtils.isFullColumnIndexing(idx))
					return new OpNormal(hop, RewriteCompressedReblock.satisfiesSizeConstraintsForCompression(hop));
				else
					return new OpDecompressing(hop);
			}
			else if(HopRewriteUtils.isBinaryMatrixMatrixOperation(hop))
				return new OpDecompressing(hop);

			// if the output size also qualifies for compression, we propagate this status
			return new OpNormal(hop, RewriteCompressedReblock.satisfiesSizeConstraintsForCompression(hop));
		}
		else
			return new OpNormal(hop, false);
	}
}
