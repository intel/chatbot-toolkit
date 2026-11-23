/*
* Copyright (C) 2024-2026 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"go.opentelemetry.io/otel"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

// TestInitializeRoutesRecordsHTTPRoute verifies that requests through the
// otelhttp-wrapped routes carry the http.route attribute the router dashboards
// group by. otelhttp takes it from the ServeMux pattern.
func TestInitializeRoutesRecordsHTTPRoute(t *testing.T) {
	reader := sdkmetric.NewManualReader()
	prev := otel.GetMeterProvider()
	otel.SetMeterProvider(sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader)))
	defer otel.SetMeterProvider(prev)

	rec := httptest.NewRecorder()
	initializeRoutes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /metrics returned %d", rec.Code)
	}

	var rm metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &rm); err != nil {
		t.Fatalf("collect failed: %v", err)
	}
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			h, ok := m.Data.(metricdata.Histogram[float64])
			if m.Name != "http.server.request.duration" || !ok {
				continue
			}
			for _, point := range h.DataPoints {
				if route, present := point.Attributes.Value("http.route"); present && route.AsString() == "/metrics" {
					return
				}
			}
			t.Fatalf("http.server.request.duration has no http.route=/metrics data point: %v", h.DataPoints)
		}
	}
	t.Fatal("http.server.request.duration was not recorded")
}
