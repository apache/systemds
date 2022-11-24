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


package org.apache.sysds.runtime.util;

import java.util.ArrayList;
import java.util.Random;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;


class LinearRegression {
	private final double intercept;
	private final double coef;

	public LinearRegression(double[] x, double[] y) {
		int n = x.length;

		double sum_x = 0.0;
		double sum_y = 0.0;

		double xx = 0.0;
		double yy = 0.0;

		for (int i = 0; i < n; i++) {
			sum_x  += x[i];
			sum_y  += y[i];
		}

		double x_tmp = sum_x / n;
		double y_tmp = sum_y / n;

		for (int i = 0; i < n; i++) {
			xx += (x[i] - x_tmp) * (x[i] - x_tmp);
			yy += (x[i] - x_tmp) * (y[i] - y_tmp);
		}

		coef = yy / xx;
		intercept = y_tmp - coef * x_tmp;
	}

	public double intercept() {
		return intercept;
	}

	public double coef() {
		return coef;
	}

}


public class EMAUtils {

	public static FrameBlock exponentialMovingAverageImputation(FrameBlock block, int search_iterations, String mode,
		int freq, Double alpha, Double beta, Double gamma) {
		int cols = block.getNumColumns();
		int rows = block.getNumRows();

		ArrayList<Double[]> data_list = new ArrayList<>();

		for (int j = 0; j < cols; j++) {
			String[] values = (String[]) block.getColumnData(j);
			Double[] data = new Double[values.length];
			for (int i = 0; i< values.length; i++) data[i] = Double.valueOf(values[i]);
			Container best_cont = new Container(new Double[]{.0}, 1000);

			Random rand = new Random();
			Container lst = null;

			for (int i = 0; i < search_iterations; i++) {
				if (Double.isNaN(alpha))
					alpha = rand.nextDouble();
				if (Double.isNaN(beta))
					beta = rand.nextDouble();
				if (Double.isNaN(gamma))
					gamma = rand.nextDouble();

				if (mode.equals("single")) {
					lst = single_exponential_smoothing(data, alpha);
				} else if (mode.equals("double")) {
					lst = double_exponential_smoothing(data, alpha, beta);
				} else if (mode.equals("triple")) {
					lst = triple_exponential_smoothing(data, alpha, beta, gamma, freq);
				}

				if(lst == null)
					throw new DMLRuntimeException("invalid null pointer");
				else if (i == 0 || lst.rsme < best_cont.rsme) {
					best_cont = lst;
					data_list.add(best_cont.values);
				}

			}
		}

		Double[][] values = new Double[][]{};
		FrameBlock new_block = generateBlock(rows, cols, data_list.toArray(values));
		return new_block;
	}

	private static FrameBlock generateBlock(int rows, int cols, Double[][] values)
	{
		Types.ValueType[] schema = new Types.ValueType[cols];
		for(int i = 0; i < cols; i++) {
			schema[i] = Types.ValueType.FP64;
		}

		String[] names = new String[cols];
		for(int i = 0; i < cols; i++)
			names[i] = schema[i].toString();
		FrameBlock frameBlock = new FrameBlock(schema, names);
		frameBlock.ensureAllocatedColumns(rows);
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				frameBlock.set(row, col, values[col][row]);
		return frameBlock;
	}

	static class Container {
		public Container(Double[] vals, double error) {
			values = vals;
			rsme = error;
		}
		Double[] values;
		double rsme;
	}

	public static Container single_exponential_smoothing(Double[] data, Double alpha) {
		int n = data.length;

		Double[] pred = new Double[n];
		pred[0] = data[0];

		double val = 0;

		ArrayList<Double> not_missing = new ArrayList<>();
		ArrayList<Double> not_missing_pred = new ArrayList<>();
		int n_size = 0;

		for (int i = 1; i < n; i++) {
			if (Double.isNaN(data[i])) {
				val = pred[i - 1];
			} else {
				val = data[i];
			}

			pred[i] = alpha * val + (1 - alpha) * pred[i - 1];
		}

		for (int i = 0; i < data.length; i++) {
			if (!Double.isNaN(data[i])) {
				not_missing.add(data[i]);
				not_missing_pred.add(pred[i]);
				n_size++;
			}
		}

		double sum = .0;
		for (int i = 0; i < not_missing.size(); i++) {
			sum += Math.pow(not_missing.get(i) - not_missing_pred.get(i), 2);
		}

		double rmse = Math.sqrt(sum / n_size);

		return new Container(pred, rmse);
	}

	public static Container double_exponential_smoothing(Double[] data, Double alpha, Double beta) {
		int n = data.length;

		ArrayList<Double> pred = new ArrayList<>(n-1);
		Double[] s = new Double[n-1];
		Double[] b = new Double[n-1];

		s[0] = data[1];
		b[0] = data[1] - data[0];
		pred.add(s[0] + b[0]);

		double val = 0;

		ArrayList<Double> not_missing = new ArrayList<>();
		ArrayList<Double> not_missing_pred = new ArrayList<>();
		int n_size = 0;

		for (int i = 1; i < n-1; i++) {
			if (Double.isNaN(data[i + 1])) {
				val = pred.get(i - 1);
			} else {
				val = data[i+1];
			}

			s[i] = alpha * val + (1 - alpha) * (s[i-1] + b[i-1]);
			b[i] = beta * (s[i] - s[i-1]) + (1 - beta) * b[i-1];
			pred.add(s[i] + b[i]);
		}

		pred.add(0, data[0]);

		for (int i = 0; i < data.length; i++) {
			if (!Double.isNaN(data[i])) {
				not_missing.add(data[i]);
				not_missing_pred.add(pred.get(i));
				n_size++;
			}
		}

		double sum = .0;
		for (int i = 0; i < not_missing.size(); i++) {
			sum += Math.pow(not_missing.get(i) - not_missing_pred.get(i), 2);
		}

		double rmse = Math.sqrt(sum / n_size);
		Double[] content = new Double[pred.size()];
		return new Container(pred.toArray(content), rmse);
	}

	public static Container triple_exponential_smoothing(Double[] data, Double alpha, Double beta,
		Double gamma, Integer freq) {
		double l = freq * 2;
		ArrayList<Double> start_data = new ArrayList<>();

		for (int i = 0; i < l; i++) {
			start_data.add(data[i]);
		}

		ArrayList<Double> filt = new ArrayList<>();
		ArrayList<Double> trend = new ArrayList<>();

		double len = freq;

		if (freq % 2 == 0) {
			len = freq - 1;
		}

		for (int i = 0; i < len; i++) {
			filt.add(1./freq);
		}

		if (freq % 2 == 0) {
			filt.add(0, 0.5/freq);
			filt.add(0.5/freq);
		}

		double trend_len = l - filt.size() + 1;

		for (int i = 0; i < l - trend_len; i++) {
			double sum = 0;
			for (int j = i; j < i + filt.size(); j++) {
				sum += start_data.get(j) * filt.get(j - i);
			}

			trend.add(sum);
		}

		int cut = (int) (l - trend.size()) / 2;

		ArrayList<Double> season_tmp = new ArrayList<>();

		for (int i = cut; i < start_data.size() -  cut; i++) {
			season_tmp.add(start_data.get(i) - trend.get(i-cut));
		}

		Double[] season = new Double[freq];

		for (int i = 0; i < freq; i++) {
			double combined = 0;
			if (i + freq < trend.size()) {
				combined = (season_tmp.get(i) + season_tmp.get(i + freq)) / 2;
			} else {
				combined = season_tmp.get(i);
			}

			season[(i + (freq / 2)) % freq] = combined;
		}

		double sum = 0;
		for (int i = 0; i < season.length; i++) {
			sum += season[i];
		}

		double mean = sum / season.length;

		for (int i = 0; i < season.length; i++) {
			season[i] = season[i] - mean;
		}

		double[] x = new double[trend.size()];
		double[] y = new double[trend.size()];

		for (int i = 0; i < trend.size(); i++) {
			x[i] = i + 1;
			y[i] = trend.get(i);
		}

		LinearRegression linreg = new LinearRegression(x, y);

		int n = data.length;

		double[] s = new double[n - freq];
		s[0] = linreg.intercept();

		double[] b = new double[n - freq];
		b[0] = linreg.coef();

		double[] c = new double[n];

		for (int i = 0; i < freq; i++) {
			c[i] =  season[i];
		}

		ArrayList<Double> pred = new ArrayList<>();
		pred.add(s[0] + b[0] + c[0]);

		double val = 0;

		ArrayList<Double> not_missing = new ArrayList<>();
		ArrayList<Double> not_missing_pred = new ArrayList<>();

		int n_size = 0;

		for (int i = 1; i < n - freq; i++) {
			if (Double.isNaN(data[i + freq - 1] )) {
				val = pred.get(i - 1);
			} else {
				val = data[i + freq - 1];
			}

			s[i] = alpha * (val - c[i-1]) + (1 - alpha) * (s[i-1] + b[i-1]);
			b[i] = beta * (s[i] - s[i-1]) + (1 - beta) * b[i-1];
			c[i+freq-1] = gamma * (val - s[i]) + (1 - gamma) * c[i-1];

			pred.add(s[i] + b[i] + c[i]);
		}

		for (int i = 0; i < freq; i++) {
			pred.add(i, data[i]);
		}

		for (int i = 0; i < data.length; i++) {
			if (!Double.isNaN(data[i])) {
				not_missing.add(data[i]);
				not_missing_pred.add(pred.get(i));
				n_size++;
			}
		}

		sum = .0;
		for (int i = 0; i < not_missing.size(); i++) {
			sum += Math.pow(not_missing.get(i) - not_missing_pred.get(i), 2);
		}

		double rmse = Math.sqrt(sum / n_size);
		Double[] content = new Double[pred.size()];
		return new Container(pred.toArray(content), rmse);
	}
}
