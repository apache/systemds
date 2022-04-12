package org.apache.sysds.runtime.controlprogram.federated.monitoring.controllers;

import static io.netty.handler.codec.http.HttpResponseStatus.OK;
import static io.netty.handler.codec.http.HttpVersion.HTTP_1_1;

import io.netty.buffer.Unpooled;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.Request;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.Response;

public class CoordinatorController implements BaseController {
    @Override
    public FullHttpResponse create(Request request) {
        return null;
    }

    @Override
    public FullHttpResponse update(Request request, Long objectId) {
        return null;
    }

    @Override
    public FullHttpResponse delete(Request request, Long objectId) {
        return null;
    }

    @Override
    public FullHttpResponse get(Request request, Long objectId) {
        return Response.ok("Success");
    }

    @Override
    public FullHttpResponse getAll(Request request) {
        return Response.ok("Success");
    }
}
