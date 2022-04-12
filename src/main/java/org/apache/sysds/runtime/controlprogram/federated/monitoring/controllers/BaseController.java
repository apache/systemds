package org.apache.sysds.runtime.controlprogram.federated.monitoring.controllers;

import io.netty.handler.codec.http.FullHttpResponse;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.Request;

public interface BaseController {

    FullHttpResponse create(final Request request);

    FullHttpResponse update(final Request request, final Long objectId);

    FullHttpResponse delete(final Request request, final Long objectId);

    FullHttpResponse get(final Request request, final Long objectId);

    FullHttpResponse getAll(final Request request);
}
