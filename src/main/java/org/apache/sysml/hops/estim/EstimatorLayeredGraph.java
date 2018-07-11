package org.apache.sysml.hops.estim;

import java.lang.reflect.Array;
import java.lang.Object;
import org.apache.commons.math3.distribution.AbstractRealDistribution;
import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.apache.commons.math3.random.Well1024a;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.DenseBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.util.UtilFunctions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public class EstimatorLayeredGraph extends SparsityEstimator {

	@Override
	public double estim(MMNode root) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2){
		int layer = 3;
		int numr = 100;
		double result=0;
		LayeredGraph LGraph = new LayeredGraph(m1, m2);
		for (int h = 0; h < LGraph.nodes.size(); h++) {
			if (LGraph.nodes.get(h).getY() == 1) {
				double[] doubArray = new double[numr];
				for (int g = 0; g < numr; g++) {
					Random rand = new Random();
					//lambda is not the mean, if lambda is 2 hand in 1/2
					ExponentialDistribution random = new ExponentialDistribution(new Well1024a(), 1);
					doubArray[g] = random.sample();
				}
				LGraph.nodes.get(h).setVector(doubArray);
			}
		}
		// get r for nodes of upper layer
		for (int h = 0; h < LGraph.nodes.size(); h++) {
			if (LGraph.nodes.get(h).getY() == layer) {
				double[] ret = new double[numr];
				ret = recr(numr, LGraph.nodes.get(h));
				if(ret != null) {
					if(ret.length > 0) {
						LGraph.nodes.get(h).setVector(ret);
					}
				}
				LGraph.nodes.get(h).setValue(calculateNZ(LGraph.nodes.get(h).getVector(), numr));
			}
		}
		//calc final sparsity
		for (int h = 0; h < LGraph.nodes.size(); h++) {
			if (LGraph.nodes.get(h).getY() == layer) {
				if(LGraph.nodes.get(h).getValue()>0) {
					result=+LGraph.nodes.get(h).getValue();
				}
			}
		}
		return result/(m1.getNumRows()*m2.getNumColumns());
	}
	
	
	public double[] recr(int numr, Node tempnode) {
		if (tempnode.getInput().size() == 0) {
			if(tempnode.getY() == 1) {
				return tempnode.getVector();
			}else {
				return null;
			}
		}
		else if (tempnode.getInput().size() == 1) {
			return recr(numr, tempnode.getInput().get(0));
		} else {
			ArrayList<double[]> veclist = new ArrayList<>();
			for(int i=0; i<tempnode.getInput().size(); i++) {
				double[] ret = new double[numr];
				ret = recr(numr, tempnode.getInput().get(i));
				if(ret != null) {
					veclist.add(ret);
				}
			}
			return selfWrittenMinFunctionCauseJavaIsToDumb(veclist, numr);
		}		
	}
	
	private double[] selfWrittenMinFunctionCauseJavaIsToDumb(List<double[]> inputVectors, int vectorSize) {
		if(inputVectors.size()>0) {
			double[] minOfColumns =  new double[vectorSize];
			Arrays.fill(minOfColumns, Double.MAX_VALUE);
			for (int j=0; j < vectorSize; j++) {
				for (int i=0; i < inputVectors.size(); i++) {
					double[] inVec = inputVectors.get(i);
					minOfColumns[j] = Math.min(minOfColumns[j], inVec[j]);
				}
			}
			return minOfColumns;
		}else {
			return null;
		}
	}

	public double calculateNZ(double[] inpvec, int numr) {
		if(inpvec != null && inpvec.length > 0) {
			double nz = 0;
			double sum = 0;
			for (int i = 0; i < numr; i++) {
				sum += inpvec[i ];
			}
			nz = (numr - 1) / (sum);
			return nz;
		}
		else{
			return 0;
		}
	}

	@Override
	public double estim(MatrixCharacteristics mc1, MatrixCharacteristics mc2) {
		// TODO Auto-generated method stub
		return 0;
	}

	private class LayeredGraph {
		List<Node> nodes = new ArrayList<>();

		public LayeredGraph(MatrixBlock m1, MatrixBlock m2) {
			createNodes(m1, 1, nodes);
			createNodes(m2, 2, nodes);
		}
	}

	public void createNodes(MatrixBlock m, int mpos, List<Node> nodes) {
		Node nodeout = null;
		Node nodein = null;
		if (!m.isEmpty()) {
			if (m.getSparsity() > 0) {
				for (int i = 0; i < m.getNumRows(); i++) {
					for (int j = 0; j < m.getNumColumns(); j++) {
						if (m.getValue(i, j) > 0) {
							boolean alreadyExists = false;
							boolean alreadyExists2 = false;
							for (int k = 0; k < nodes.size(); k++) {
								if (nodes.get(k).getX() == i && nodes.get(k).getY() == mpos) {
									alreadyExists = true;
								}
							}
							if (!alreadyExists) {
								nodein = new Node(i, mpos);
								nodes.add(nodein);
							} else {
								for (int k = 0; k < nodes.size(); k++) {
									if (nodes.get(k).getX() == i && nodes.get(k).getY() == mpos) {
									 nodein = nodes.get(k);
									}
								}
							}
							for (int k = 0; k < nodes.size(); k++) {
								if (nodes.get(k).getX() == j && nodes.get(k).getY() == mpos + 1) {
									alreadyExists2 = true;
								}
							}
							if (!alreadyExists2) {
								nodeout = new Node(j, mpos + 1);
								nodes.add(nodeout);

							} else {
								for (int k = 0; k < nodes.size(); k++) {
									if (nodes.get(k).getX() == j && nodes.get(k).getY() == mpos + 1) {
										nodeout = nodes.get(k);
									}
								}
							}
							// naming is weird here but had to change way of
							// propagation when it was already implemented
							nodeout.addnz(nodein);
						}
					}
				}
			}
		}
	}

	private static class Node {
		int xpos;
		int ypos;
		double[] r_vector;
		List<Node> input = new ArrayList<>();
		double value;

		public void setValue(double inp) {
			value = inp;
		}

		public double getValue() {
			return value;
		}

		public Node(int x, int y) {
			xpos = x;
			ypos = y;
		}

		public List<Node> getInput() {
			return input;
		}

		public int getX() {
			return xpos;
		}

		public int getY() {
			return ypos;
		}

		public double[] getVector() {
			return r_vector;
		}

		public void setVector(double[] r_input) {
			r_vector = r_input;
		}

		public void addnz(Node dest) {
			input.add(dest);
		}
	}
}
