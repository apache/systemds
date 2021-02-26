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

import com.google.common.collect.ArrayTable;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.lang.Math;

public class EMAUtils {

	public static FrameBlock exponentialMovingAverageImputation(FrameBlock data2, int search_iterations, String mode, int freq) {
		Double[] data= new Double[]{.112,.118,.132,.129, null,.135,.148,.148,null,.119,.104,.118,.115,.126,.141,.135,.125,.149,.170,.170,null,.133,null,.140,.145,.150,.178,.163,.172,.178,.199,.199,.184,.162,.146,.166,.171,.180,.193,.181,.183,.218,.230,.242,.209,.191,.172,.194,.196,.196,.236,.235,.229,.243,.264,.272,.237,.211,.180,.201,.204,.188,.235,.227,.234,null,.302,.293,.259,.229,.203,.229,.242,.233,.267,.269,.270,.315,.364,.347,.312,.274,.237,.278,.284,.277,null,null,null,.374,.413,.405,.355,.306,.271,.306,.315,.301,.356,.348,.355,null,.465,.467,.404,.347,null,.336,.340,.318,null,.348,.363,.435,.491,.505,.404,.359,.310,.337,.360,.342,.406,.396,.420,.472,.548,.559,.463,.407,.362,null,.417,.391,.419,.461,null,.535,.622,.606,.508,.461,.390,.432};

		data = new Double[]{41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015, 31.1802, 37.7179, 40.4202,
				51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668, 35.1919, 44.3197, 47.9137};
		int n = data.length;

		double best_alpha = .0;
		double best_beta = .0;
		double best_gamma = .0;
		Container best_cont = new Container(new Double[]{.0}, 1000);

		Random rand = new Random();
		Container lst = null;

		for (int i = 0; i < search_iterations; i++) {
			Double alpha = rand.nextDouble();
			Double beta = rand.nextDouble();
			Double gamma = rand.nextDouble();

			mode = "triple";

			if (mode.equals("single")) {
				lst = single_exponential_smoothing(data, alpha);
			} else if (mode.equals("double")) {
				lst = double_exponential_smoothing(data, alpha, beta);
			} else if (mode.equals("triple")) {
				lst = triple_exponential_smoothing(data, alpha, beta, gamma, freq);
			}

			if (i == 0 || lst.rsme < best_cont.rsme) {
				best_cont = lst;
				best_alpha = alpha;
				best_beta = beta;
				best_gamma = gamma;
			}

		}

		return null;
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
		ArrayList<Double> not_missing_sq = new ArrayList<>();
		int n_size = 0;

		for (int i = 1; i < n; i++) {
			if (data[i] == null) {
				val = pred[i - 1];
			} else {
				val = data[i];
				not_missing.add(val);
				not_missing_sq.add(Math.pow(val, 2));
				n_size++;
			}

			pred[i] = alpha * val + (1 - alpha) * pred[i - 1];
		}

		double sum = .0;
		for (int i = 0; i < not_missing.size(); i++) {
			sum += not_missing.get(i) - not_missing_sq.get(i);
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
		ArrayList<Double> not_missing_sq = new ArrayList<>();
		int n_size = 0;

		for (int i = 1; i < n-1; i++) {
			if (data[i+1] == null) {
				val = pred.get(i - 1);
			} else {
				val = data[i+1];
				not_missing.add(val);
				not_missing_sq.add(Math.pow(val, 2));
				n_size++;
			}

			s[i] = alpha * val + (1 - alpha) * (s[i-1] + b[i-1]);
			b[i] = beta * (s[i] - s[i-1]) + (1 - beta) * b[i-1];
			pred.add(s[i] + b[i]);
		}

		double sum = .0;
		for (int i = 0; i < not_missing.size(); i++) {
			sum += not_missing.get(i) - not_missing_sq.get(i);
		}

		pred.add(0, data[0]);
		double rmse = Math.sqrt(sum / n_size);
		Double[] content = new Double[pred.size()];
		return new Container(pred.toArray(content), rmse);
	}

	public static ArrayList<Double> multiply(ArrayList<Double> array_1, ArrayList<Double> array_2) {
		ArrayList<Double> new_list = new ArrayList<>();
		for (int i = 0; i < array_1.size(); i++) {
			new_list.add(array_1.get(i) * array_2.get(i));
		}

		return new_list;
	}

	public static Container triple_exponential_smoothing(Double[] data, Double alpha, Double beta, Double gamma, Integer freq) {
		alpha = 0.5;
		beta = 0.5;
		gamma = 0.5;
		freq = 4;

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
			filt.add(0.25);
		}

		if (freq % 2 == 0) {
			filt.add(0, 0.125);
			filt.add(0.125);
		}

		double trend_len = l - filt.size() + 1;

		for (int i = 0; i < l - trend_len; i++) {
			double sum = 0;
			for (int j = i; j < i + filt.size(); j++) {
				sum += data[j] * filt.get(j - i);
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

		return null;
	}
}
