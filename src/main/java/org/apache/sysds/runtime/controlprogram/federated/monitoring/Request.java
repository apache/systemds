package org.apache.sysds.runtime.controlprogram.federated.monitoring;

import io.netty.handler.codec.http.HttpRequest;

public class Request {
    private HttpRequest _context;
    private String _body;

    public HttpRequest getContext() {
        return _context;
    }

    public void setContext(final HttpRequest requestContext) {
        this._context = requestContext;
    }

    public String getBody() {
        return _body;
    }

    public void setBody(final String content) {
        this._body = content;
    }
}
