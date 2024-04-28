package org.apache.sysds.api.ropt.strategies;

import org.apache.sysds.api.ropt.SearchSpace;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class GridSearch extends SearchStrategy {
    private int _numberExecutors;
    private Iterator<SearchSpace.InstanceType> _executorTypesIterator;
    private SearchSpace.InstanceType _currentExecutorType;
    private Iterator<SearchSpace.InstanceSize> _executorSizesIterator = null;
    private SearchSpace.InstanceSize _currentExecutorSize;
    private SearchSpace.InstanceType _currentDriverType;
    private final Iterator<SearchSpace.InstanceType> _driverTypesIterator;
    private SearchSpace.InstanceSize _currentDriverSize;
    private Iterator<SearchSpace.InstanceSize> _driverSizesIterator = null;

    public GridSearch(SearchSpace searchSpace) {
        super(searchSpace);
        _numberExecutors = SearchSpace.MIN_EXECUTORS;
        _driverTypesIterator = _searchSpace.getInstanceTypeDomainDriver().iterator();
        _hasNext = getNextInstanceCombinationDriver();
        if (!_hasNext)
            throw new RuntimeException("No instance type for driver to parse");

        initExecutorIterators();
    }

    private void initExecutorIterators() {
        _executorTypesIterator = _searchSpace.getInstanceTypeDomainExecutors().iterator();
        _hasNext = getNextInstanceCombinationExecutor();
        if (!_hasNext)
            throw new RuntimeException("No instance type for executor to parse");
    }

    private boolean getNextInstanceCombinationExecutor() {
        // parse first all the sizes of the current type
        if (_executorSizesIterator != null && getNextInstanceSizeExecutor())
            return true;
        // else parse the next type
        if (getNextInstanceTypeExecutor()) {
            // init the size iterator for the newly parsed type
            _executorSizesIterator = _searchSpace.getInstanceSizeDomainExecutors().get(_currentExecutorType).iterator();
            // and update the current size
            _hasNext = getNextInstanceSizeExecutor();
            if (!_hasNext)
                throw new RuntimeException("Error at initializing search space for executor");
            return true;
        }
        return false;
    }

    private boolean getNextInstanceTypeExecutor() {
        if (_executorTypesIterator.hasNext()) {
            _currentExecutorType = _executorTypesIterator.next();
            return true;
        } else {
            _currentExecutorType = null;
            return false;
        }
    }

    private boolean getNextInstanceSizeExecutor() {
        if (_executorSizesIterator.hasNext()) {
            _currentExecutorSize = _executorSizesIterator.next();
            return true;
        } else {
            _currentExecutorSize = null;
            return false;
        }
    }

    private boolean getNextInstanceCombinationDriver() {
        // parse first all the sizes of the current type
        if (_driverSizesIterator != null && getNextInstanceSizeDriver())
            return true;
        // else parse the next type
        if (getNextInstanceTypeDriver()) {
            // init the size iterator for the newly parsed type
            _driverSizesIterator = _searchSpace.getInstanceSizeDomainDriver().get(_currentDriverType).iterator();
            // and update the current size
            _hasNext = getNextInstanceSizeDriver();
            if (!_hasNext)
                throw new RuntimeException("Error at initializing search space for driver");
            return true;
        }
        return false;
    }

    private boolean getNextInstanceTypeDriver() {
        if (_driverTypesIterator.hasNext()) {
            _currentDriverType = _driverTypesIterator.next();
            return true;
        } else {
            _currentDriverType = null;
            return false;
        }
    }

    private boolean getNextInstanceSizeDriver() {
        if (_driverSizesIterator.hasNext()) {
            _currentDriverSize = _driverSizesIterator.next();
            return true;
        } else {
            _currentDriverSize = null;
            return false;
        }
    }

    @Override
    public SearchSpace.SearchPoint enumerateNext() {
        if (!_hasNext)
            throw new NoSuchElementException("Iterator is already exhausted");

        SearchSpace.SearchPoint nextPoint;
        if (_numberExecutors == 0) {
            nextPoint = new SearchSpace.SearchPoint(null, null, _currentDriverType, _currentDriverSize, _numberExecutors);
        } else {
            nextPoint = new SearchSpace.SearchPoint(_currentExecutorType, _currentExecutorSize, _currentDriverType, _currentDriverSize, _numberExecutors);
        }

        // prepare next
        // parse all possible number of executors first (most inner iteration)
        if (_numberExecutors < SearchSpace.MAX_EXECUTORS) {
            _numberExecutors++;
            return nextPoint;
        } else {
            _numberExecutors = SearchSpace.MIN_EXECUTORS;
        }
        // parse all possible combinations for executor instances for a certain driver instance
        _hasNext = getNextInstanceCombinationExecutor();
        if (!_hasNext) {
            // once all executor combinations are exhausted get next driver combination
            _hasNext = getNextInstanceCombinationDriver();
            if (_hasNext) {
                // and then reinitialize the executor iterators
                initExecutorIterators();
            }
        }

        return nextPoint;
    }
}
