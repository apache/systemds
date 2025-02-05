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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.utils.Explain;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class DependencyThreadPool {

	protected static final Log LOG = LogFactory.getLog(DependencyThreadPool.class.getName());
	private final ExecutorService _pool;

	public DependencyThreadPool(int k) {
		_pool = CommonThreadPool.get(k);
	}

	public void shutdown() {
		_pool.shutdown();
	}

	public List<Future<Future<?>>> submitAll(List<DependencyTask<?>> dtasks) {
		List<Future<Future<?>>> futures = new ArrayList<>();
		List<Integer> rdyTasks = new ArrayList<>();
		int i = 0;
		// sort by priority
		Collections.sort(dtasks);
		for(DependencyTask<?> t : dtasks) {
			CompletableFuture<Future<?>> f = new CompletableFuture<>();
			t.addPool(_pool);
			if(!t.isReady()) {
				t.assignFuture(f);
			}
			else {
				// need to save rdy tasks before execution begins otherwise tasks may start 2 times
				rdyTasks.add(i);
			}
			futures.add(f);
			i++;
		}
		LOG.debug("Initial Starting tasks: \n\t" +
				rdyTasks.stream().map(index -> dtasks.get(index).toString()).collect(Collectors.joining("\n\t")));
		// Two stages to avoid race condition!
		for(Integer index : rdyTasks) {
			synchronized(_pool) {
				((CompletableFuture<Future<?>>) futures.get(index)).complete(_pool.submit(dtasks.get(index)));
			}

		}
		return futures;
	}

	public List<Future<Future<?>>> submitAll(List<? extends Callable<?>> tasks,
		List<List<? extends Callable<?>>> dependencies) {
		List<DependencyTask<?>> dtasks = createDependencyTasks(tasks, dependencies);
		return submitAll(dtasks);
	}

	public List<Object> submitAllAndWait(List<DependencyTask<?>> dtasks)
		throws ExecutionException, InterruptedException {
		List<Object> res = new ArrayList<>();
		if(LOG.isDebugEnabled()) {
			if (dtasks != null && dtasks.size() > 0)
				explainTaskGraph(dtasks);
		}
		List<Future<Future<?>>> futures = submitAll(dtasks);
		int i = 0;
		for(Future<Future<?>> ff : futures) {
			if(dtasks.get(i) instanceof DependencyWrapperTask) {
				for(Future<Future<?>> f : ((DependencyWrapperTask<?>) dtasks.get(i)).getWrappedTaskFuture()) {
					res.add(f.get().get());
				}
			}
			else {
				res.add(ff.get().get());
			}
			i++;
		}
		return res;
	}

	public static DependencyTask<?> createDependencyTask(Callable<?> task) {
		return new DependencyTask<>(task, new ArrayList<>());
	}

	/*
	 * Creates the Dependency list from a map and the tasks. The map specifies which tasks 
	 * should have a Dependency on which other task. e.g.
	 * ([0, 3], [4, 6])   means the 1st 3 tasks in the list are dependent on tasks at index 4 and 5
	 * ([-2, -1], [0, 5]) means the last task depends on the first 5 tasks.
	 * ([dependent start index, dependent end index (excluding)], 
	 *  [parent start index, parent end index (excluding)])
	 */
	public static List<List<? extends Callable<?>>> createDependencyList(List<? extends Callable<?>> tasks,
		Map<Integer[], Integer[]> depMap, List<List<? extends Callable<?>>> dep) {
		if(depMap != null) {
			depMap.forEach((ti, di) -> {
				ti[0] = ti[0] < 0 ? dep.size() + ti[0] + 1 : ti[0];
				ti[1] = ti[1] < 0 ? dep.size() + ti[1] + 1 : ti[1];
				di[0] = di[0] < 0 ? tasks.size() + di[0] + 1 : di[0];
				di[1] = di[1] < 0 ? tasks.size() + di[1] + 1 : di[1];
				for(int r = ti[0]; r < ti[1]; r++) {
					if(dep.get(r) == null)
						dep.set(r, tasks.subList(di[0], di[1]));
					else
						dep.set(r, Stream.concat(dep.get(r).stream(), tasks.subList(di[0], di[1]).stream())
							.collect(Collectors.toList()));
				}
			});
		}
		return dep;
	}

	public static List<DependencyTask<?>> createDependencyTasks(List<? extends Callable<?>> tasks,
		List<List<? extends Callable<?>>> dependencies) {
		if(dependencies != null && tasks.size() != dependencies.size())
			throw new DMLRuntimeException(
				"Could not create DependencyTasks since the input array sizes are mismatching");
		List<DependencyTask<?>> ret = new ArrayList<>();
		Map<Callable<?>, DependencyTask<?>> map = new HashMap<>();
		for(Callable<?> task : tasks) {
			DependencyTask<?> dt;
			if(task instanceof DependencyTask) {
				dt = (DependencyTask<?>) task;
			}
			else {
				dt = new DependencyTask<>(task, new ArrayList<>());
			}
			ret.add(dt);
			map.put(task, dt);
		}
		if(dependencies == null)
			return ret;

		for(int i = 0; i < tasks.size(); i++) {
			List<? extends Callable<?>> deps = dependencies.get(i);
			if(deps == null)
				continue;
			DependencyTask<?> t = ret.get(i);
			for(Callable<?> dep : deps) {
				DependencyTask<?> dt = map.get(dep);
				if(LOG.isDebugEnabled()) {
					t._dependencyTasks = t._dependencyTasks == null ? new ArrayList<>() : t._dependencyTasks;
					t._dependencyTasks.add(dt);
				}
				if(dt != null)
					dt.addDependent(t);
			}
		}
		return ret;
	}

	/*
	 * Prints the task-graph level-wise, however, the printed
	 * output doesn't specify which task of level l depends
	 * on which task of level (l-1).
	 */
	public static void explainTaskGraph(List<DependencyTask<?>> tasks) {
		Map<DependencyTask<?>, Integer> levelMap = new HashMap<>();
		int depth = 1;
		while (levelMap.size() < tasks.size()) {
			for (int i=0; i<tasks.size(); i++) {
				DependencyTask<?> dt = tasks.get(i);
				if (dt._dependencyTasks == null || dt._dependencyTasks.size() == 0)
					levelMap.put(dt, 0);
				if (dt._dependencyTasks != null) {
					List<DependencyTask<?>> parents = dt._dependencyTasks;
					int[] parentLevels = new int[parents.size()];
					boolean missing = false;
					for (int p=0; p<parents.size(); p++) {
						if (!levelMap.containsKey(parents.get(p)))
							missing = true;
						else
							parentLevels[p] = levelMap.get(parents.get(p));
					}
					if (missing)
						continue;
					int maxParentLevel = Arrays.stream(parentLevels).max().getAsInt();
					levelMap.put(dt, maxParentLevel+1);
					if (maxParentLevel+1 == depth)
						depth++;
				}
			}
		}
		StringBuilder sbs[] = new StringBuilder[depth];
		String offsets[] = new String[depth];
		for (Map.Entry<DependencyTask<?>, Integer> entry : levelMap.entrySet()) {
			int level = entry.getValue();
			if (sbs[level] == null) {
				sbs[level] = new StringBuilder();
				offsets[level] = Explain.createOffset(level);
			}
			sbs[level].append(offsets[level]);
			sbs[level].append(entry.getKey().toString()+"\n");
		}
		StringBuilder sb = new StringBuilder("\n");
		sb.append("EXPlAIN (TASK-GRAPH):");
		for (int i=0; i<sbs.length; i++) {
			sb.append(sbs[i].toString());
		}
		LOG.debug(sb.toString());

	}
}
