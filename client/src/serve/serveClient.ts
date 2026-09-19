/**
 * Typed client for the spikeforge-serve REST and stream API.
 *
 * The request/response types are generated from `protocol/serve/*.schema.json`
 * into `client/src/protocol/generated.ts`, so the client and the service
 * cannot drift: regenerate with `npm run gen:protocol` after editing a schema.
 * This module adds only the thin transport wrapper over those types.
 */

import type {
  BundleInfo,
  PredictRequest,
  PredictResponse,
  ResetResponse,
  StreamMessage,
} from "../protocol/generated";

/** Raised when the service answers a non-2xx status. */
export class ServeError extends Error {
  readonly status: number;
  readonly type: string;

  constructor(status: number, type: string, message: string) {
    super(`HTTP ${status} ${type}: ${message}`);
    this.name = "ServeError";
    this.status = status;
    this.type = type;
  }
}

/** Options for {@link ServeClient}. */
export interface ServeClientOptions {
  baseUrl: string;
  token?: string;
  fetchImpl?: typeof fetch;
}

/** The typed error body the service returns. */
interface ServeErrorBody {
  error?: { type?: string; message?: string };
  detail?: string;
}

/** Return the `{type, message}` an error body carries, with a fallback. */
function errorParts(body: unknown): { type: string; message: string } {
  const parsed = (body ?? {}) as ServeErrorBody;
  if (parsed.error && typeof parsed.error.message === "string") {
    return {
      type: parsed.error.type ?? "error",
      message: parsed.error.message,
    };
  }
  if (typeof parsed.detail === "string") {
    return { type: "http_error", message: parsed.detail };
  }
  return { type: "http_error", message: "service returned an error status" };
}

/** A typed client for a running spikeforge-serve service. */
export class ServeClient {
  private readonly baseUrl: string;
  private readonly token?: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ServeClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.token = options.token;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  private async request<T>(path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = {};
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
    }
    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }
    const response = await this.fetchImpl(this.baseUrl + path, {
      method: body === undefined ? "GET" : "POST",
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const payload: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const { type, message } = errorParts(payload);
      throw new ServeError(response.status, type, message);
    }
    return payload as T;
  }

  /** Return the service's liveness payload. */
  async health(): Promise<{ status: string }> {
    return this.request<{ status: string }>("/health");
  }

  /** Return the service's readiness payload. */
  async ready(): Promise<{ status: string }> {
    return this.request<{ status: string }>("/ready");
  }

  /** Return the loaded bundle's self-describing metadata. */
  async bundleInfo(): Promise<BundleInfo> {
    return this.request<BundleInfo>("/v1/bundle");
  }

  /** Advance a session over the request's frames; one prediction per frame. */
  async predict(request: PredictRequest): Promise<PredictResponse> {
    return this.request<PredictResponse>("/v1/predict", request);
  }

  /** Clear a session's temporal state and return its step count. */
  async reset(sessionId?: string): Promise<ResetResponse> {
    const body = sessionId ? { session_id: sessionId } : {};
    return this.request<ResetResponse>("/v1/reset", body);
  }

  /** Return the WebSocket URL for the streaming endpoint. */
  streamUrl(): string {
    const secure = this.baseUrl.startsWith("https://");
    const host = this.baseUrl.replace(/^https?:\/\//, "");
    return `${secure ? "wss://" : "ws://"}${host}/v1/stream`;
  }
}

/** Handlers for a streaming session. */
export interface StreamHandlers {
  onMessage: (message: StreamMessage) => void;
  onError?: (error: Event) => void;
  onClose?: () => void;
}

/** Send `messages` over the stream, one reply per request message. */
export function sendFrames(
  socket: WebSocket,
  messages: StreamMessage[]
): void {
  for (const message of messages) {
    socket.send(JSON.stringify(message));
  }
}

/** Open the streaming endpoint and dispatch typed replies to `handlers`. */
export function openStream(
  client: ServeClient,
  handlers: StreamHandlers
): WebSocket {
  const socket = new WebSocket(client.streamUrl());
  socket.onmessage = (event: MessageEvent) => {
    handlers.onMessage(JSON.parse(String(event.data)) as StreamMessage);
  };
  if (handlers.onError) {
    const onError = handlers.onError;
    socket.onerror = (event) => onError(event);
  }
  if (handlers.onClose) {
    const onClose = handlers.onClose;
    socket.onclose = () => onClose();
  }
  return socket;
}
