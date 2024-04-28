package org.apache.sysds.api.ropt.strategies;

import org.apache.sysds.api.ropt.SearchSpace;

public abstract class SearchStrategy {

    enum Strategy {
        GRID_SEARCH,
        MEMORY_SEARCH,
        BAYESIAN_OPTIMIZATION
    }

    protected boolean _hasNext;
    protected final SearchSpace _searchSpace;

    SearchStrategy(SearchSpace searchSpace) {
        _searchSpace = searchSpace;
    }

    public boolean hasNext() {
        return _hasNext;
    }

    public abstract SearchSpace.SearchPoint enumerateNext();
}
