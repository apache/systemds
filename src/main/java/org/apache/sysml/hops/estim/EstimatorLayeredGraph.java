package org.apache.sysml.hops.estim;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public class EstimatorLayeredGraph extends SparsityEstimator {

	@Override
	public double estim(MMNode root) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		int numr = 10;
		LayeredGraph LGraph = new LayeredGraph(m1, m2);
		// Math.log(1-rand.nextDouble())/(-lambda);
		for (int h = 0; h < LGraph.nodes.size(); h++) {
			if (LGraph.nodes.get(h).getY() == 1) {
				double[] doubArray = new double[numr];
				for (int g = 0; g < numr; g++) {
					doubArray[g] = Math.log(1 - rand.nextDouble()) / (-lambda);
				}
				LGraph.nodes.get(h).setVector(doubArray);
			}
		}

		return 0;
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
		Node nodein = null;
		Node nodeout = null;
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

							} else {
								for (int k = 0; k < nodes.size(); k++) {
									if (nodes.get(k).getX() == j && nodes.get(k).getY() == mpos + 1) {
										nodeout = nodes.get(k);
									}
								}
							}
							nodein.addnz(nodeout);
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

		public Node(int x, int y) {
			int xpos = x;
			int ypos = y;
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
