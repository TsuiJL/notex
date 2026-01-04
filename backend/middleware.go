package backend

import (
	"bytes"
	"fmt"
	"io"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/kataras/golog"
)

// getClientIP extracts the real client IP from the request, taking into account
// proxies and load balancers that set X-Forwarded-For, X-Real-IP, etc.
func getClientIP(c *gin.Context) string {
	// Check X-Forwarded-For header (set by Nginx and other proxies)
	// Format: X-Forwarded-For: <client>, <proxy1>, <proxy2>
	// The first IP is the original client IP
	if xff := c.GetHeader("X-Forwarded-For"); xff != "" {
		// Parse the first IP from the comma-separated list
		for i, char := range xff {
			if char == ',' {
				return xff[:i]
			}
		}
		return xff
	}

	// Check X-Real-IP header (often set by Nginx)
	if xri := c.GetHeader("X-Real-IP"); xri != "" {
		return xri
	}

	// Check CF-Connecting-IP (Cloudflare)
	if cfip := c.GetHeader("CF-Connecting-IP"); cfip != "" {
		return cfip
	}

	// Check True-Client-IP (Akamai and Cloudflare Enterprise)
	if tci := c.GetHeader("True-Client-IP"); tci != "" {
		return tci
	}

	// Fall back to RemoteAddr
	return c.ClientIP()
}

// responseBodyWriter wraps gin.ResponseWriter to capture response body
type responseBodyWriter struct {
	gin.ResponseWriter
	body *bytes.Buffer
}

func (r *responseBodyWriter) Write(b []byte) (int, error) {
	r.body.Write(b)
	return r.ResponseWriter.Write(b)
}

// AuditMiddleware creates a middleware that logs all HTTP requests with full details
func AuditMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		// Capture request body for POST/PUT/PATCH requests
		var requestBody string
		if c.Request.Method == "POST" || c.Request.Method == "PUT" || c.Request.Method == "PATCH" {
			bodyBytes, err := io.ReadAll(c.Request.Body)
			if err == nil {
				c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
				if len(bodyBytes) > 1000 {
					requestBody = string(bodyBytes[:1000]) + "... (truncated)"
				} else {
					requestBody = string(bodyBytes)
				}
			}
		}

		// Capture response body
		w := &responseBodyWriter{
			ResponseWriter: c.Writer,
			body:           bytes.NewBufferString(""),
		}
		c.Writer = w

		// Process request
		c.Next()

		// Calculate latency
		latency := time.Since(start).Milliseconds()

		// Get client IP (handling proxy headers)
		clientIP := getClientIP(c)

		// Build log message
		msg := fmt.Sprintf("[AUDIT] client_ip=%s method=%s path=%s status=%d latency_ms=%d",
			clientIP, c.Request.Method, c.Request.URL.Path, c.Writer.Status(), latency)

		if requestBody != "" {
			msg += fmt.Sprintf(" request_body=%s", requestBody)
		}

		if w.body.Len() > 0 {
			respBytes := w.body.Bytes()
			if len(respBytes) > 1000 {
				msg += fmt.Sprintf(" response_body=%s... (truncated)", string(respBytes[:1000]))
			} else {
				msg += fmt.Sprintf(" response_body=%s", string(respBytes))
			}
		}

		if len(c.Errors) > 0 {
			msg += fmt.Sprintf(" errors=%s", c.Errors.String())
		}

		golog.Info(msg)
	}
}

// AuditMiddlewareLite creates a lightweight middleware that logs HTTP requests
// without capturing request/response bodies (better performance)
func AuditMiddlewareLite() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		// Process request
		c.Next()

		// Calculate latency
		latency := time.Since(start).Milliseconds()

		// Get client IP (handling proxy headers)
		clientIP := getClientIP(c)

		// Build log message
		msg := fmt.Sprintf("[AUDIT] client_ip=%s method=%s path=%s status=%d latency_ms=%d user_agent=%s",
			clientIP, c.Request.Method, c.Request.URL.Path, c.Writer.Status(), latency, c.GetHeader("User-Agent"))

		if len(c.Errors) > 0 {
			msg += fmt.Sprintf(" errors=%s", c.Errors.String())
		}

		golog.Info(msg)
	}
}