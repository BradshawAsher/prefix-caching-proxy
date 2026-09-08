pub mod radix;
pub mod tokenizer;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use parking_lot::RwLock;
use radix::{RadixTree, WorkerId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokenizer::{ChatMessage, PromptTokenizer};
use tracing::{info, warn};

/// A downstream inference worker node (e.g. Ollama or vLLM instance).
#[derive(Debug, Clone)]
pub struct WorkerNode {
    pub id: WorkerId,
    pub name: String,
    pub base_url: String,
    pub active_requests: Arc<AtomicUsize>,
}

/// Global shared state passed into Axum request handlers.
#[derive(Clone)]
pub struct AppState {
    pub tree: Arc<RwLock<RadixTree>>,
    pub tokenizer: Arc<PromptTokenizer>,
    pub workers: Vec<WorkerNode>,
    pub client: reqwest::Client,
    pub min_prefix_match_ratio: f64,
}

/// OpenAI-compatible chat completion request schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Diagnostic health & cache stats response.
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub status: String,
    pub total_prefix_nodes: usize,
    pub active_workers: Vec<WorkerStat>,
}

#[derive(Debug, Serialize)]
pub struct WorkerStat {
    pub id: WorkerId,
    pub name: String,
    pub url: String,
    pub active_connections: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    info!("Starting Prefix-Caching KV Router (MLSys Layer-7 Reverse Proxy)...");

    // 1. Initialize the Tokenizer
    let tokenizer_path = std::env::var("TOKENIZER_PATH").unwrap_or_else(|_| "tokenizer.json".to_string());
    info!("Loading BPE Tokenizer from '{}'...", tokenizer_path);
    let tokenizer = PromptTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer.json. Make sure the file exists in the project root.");

    // 2. Configure Downstream Workers (e.g. Ollama or vLLM)
    let worker1_url = std::env::var("WORKER_1_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
    let worker2_url = std::env::var("WORKER_2_URL").unwrap_or_else(|_| "http://127.0.0.1:8002".to_string());

    let workers = vec![
        WorkerNode {
            id: 1,
            name: "Worker-1".to_string(),
            base_url: worker1_url.clone(),
            active_requests: Arc::new(AtomicUsize::new(0)),
        },
        WorkerNode {
            id: 2,
            name: "Worker-2".to_string(),
            base_url: worker2_url.clone(),
            active_requests: Arc::new(AtomicUsize::new(0)),
        },
    ];

    info!("Registered 2 inference worker targets:");
    info!("  - Worker 1: {}", worker1_url);
    info!("  - Worker 2: {}", worker2_url);

    // 3. Initialize Shared Application State
    let state = AppState {
        tree: Arc::new(RwLock::new(RadixTree::new())),
        tokenizer: Arc::new(tokenizer),
        workers,
        client: reqwest::Client::builder()
            .pool_max_idle_per_host(50)
            .build()?,
        min_prefix_match_ratio: 0.20, // Require at least 20% prefix match for cache affinity
    };

    // 4. Build Axum Routes
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/stats", get(get_stats))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()
        .unwrap_or(8000);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("🚀 Prefix-Caching Proxy listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check endpoint.
async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({ "status": "healthy" })))
}

/// Real-time statistics on Radix Tree nodes and worker connection queues.
async fn get_stats(State(state): State<AppState>) -> impl IntoResponse {
    let tree = state.tree.read();
    let stats = StatsResponse {
        status: "active".to_string(),
        total_prefix_nodes: tree.node_count,
        active_workers: state
            .workers
            .iter()
            .map(|w| WorkerStat {
                id: w.id,
                name: w.name.clone(),
                url: w.base_url.clone(),
                active_connections: w.active_requests.load(Ordering::Relaxed),
            })
            .collect(),
    };
    (StatusCode::OK, Json(stats))
}

/// Primary Layer-7 Proxy Handler for /v1/chat/completions.
async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let start_time = Instant::now();

    // 1. Format chat messages into prompt string
    let prompt_text = PromptTokenizer::format_chat_messages(&payload.messages);

    // 2. Tokenize prompt into u32 token IDs (sub-100 microseconds)
    let token_ids = match state.tokenizer.encode(&prompt_text) {
        Ok(tokens) => tokens,
        Err(err) => {
            warn!("Tokenization failed: {}", err);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Tokenization failed" })),
            )
                .into_response();
        }
    };

    let total_tokens = token_ids.len();

    // 3. Query In-Memory Radix Tree for Longest Prefix Match (LPM)
    let match_result = state.tree.read().find_longest_prefix(&token_ids);

    // 4. Affinity Routing Engine (Cache Hit vs. Least Connections)
    let (chosen_worker_id, routing_reason) = if let Some(ref m) = match_result {
        let match_ratio = (m.matched_tokens as f64) / (total_tokens.max(1) as f64);
        if match_ratio >= state.min_prefix_match_ratio {
            (
                m.worker_id,
                format!("CACHE HIT: matched {}/{} tokens ({:.1}%)", m.matched_tokens, total_tokens, match_ratio * 100.0),
            )
        } else {
            // Below threshold; fall back to least connections
            let least_busy = select_least_loaded_worker(&state.workers);
            (
                least_busy.id,
                format!("CACHE WEAK: match {:.1}% < 20%; fall back to least loaded", match_ratio * 100.0),
            )
        }
    } else {
        let least_busy = select_least_loaded_worker(&state.workers);
        (least_busy.id, "CACHE MISS: 0 tokens cached; routing to least loaded".to_string())
    };

    let target_worker = state
        .workers
        .iter()
        .find(|w| w.id == chosen_worker_id)
        .cloned()
        .unwrap_or_else(|| state.workers[0].clone());

    let routing_duration_us = start_time.elapsed().as_micros();
    info!(
        "[ROUTER] -> Worker {} ({}) | {} | Overhead: {}μs",
        target_worker.id, target_worker.base_url, routing_reason, routing_duration_us
    );

    // 5. Track in-flight connections (RAII drop guard)
    target_worker.active_requests.fetch_add(1, Ordering::SeqCst);
    let active_counter = target_worker.active_requests.clone();

    // 6. Forward Request to Chosen Worker Target
    let target_url = format!("{}/v1/chat/completions", target_worker.base_url);
    let response_result = state.client.post(&target_url).json(&payload).send().await;

    // Decrement connection counter
    active_counter.fetch_sub(1, Ordering::SeqCst);

    match response_result {
        Ok(res) => {
            let status = res.status();
            let mut headers = HeaderMap::new();
            if let Some(content_type) = res.headers().get(reqwest::header::CONTENT_TYPE) {
                headers.insert(axum::http::header::CONTENT_TYPE, content_type.clone());
            }

            // 7. Update Radix Tree with the cached prompt for this worker
            let tree_tokens = token_ids.clone();
            let tree_ref = state.tree.clone();
            let worker_id = target_worker.id;

            tokio::spawn(async move {
                tree_ref.write().insert(&tree_tokens, worker_id);
            });

            // If streaming, pipe downstream SSE directly to client
            if payload.stream {
                headers.insert(
                    axum::http::header::CONTENT_TYPE,
                    HeaderValue::from_static("text/event-stream"),
                );
                headers.insert(
                    axum::http::header::CACHE_CONTROL,
                    HeaderValue::from_static("no-cache"),
                );

                let stream = res.bytes_stream();
                let body = Body::from_stream(stream);
                (status, headers, body).into_response()
            } else {
                let bytes = match res.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        return (
                            StatusCode::BAD_GATEWAY,
                            Json(serde_json::json!({ "error": format!("Worker read error: {}", e) })),
                        )
                            .into_response();
                    }
                };
                (status, headers, Body::from(bytes)).into_response()
            }
        }
        Err(err) => {
            warn!("Failed to forward request to worker {}: {}", target_worker.id, err);
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": "Downstream model worker unreachable",
                    "target": target_worker.base_url,
                    "details": err.to_string()
                })),
            )
                .into_response()
        }
    }
}

/// Helper to select the worker node with the fewest active in-flight connections.
fn select_least_loaded_worker(workers: &[WorkerNode]) -> &WorkerNode {
    workers
        .iter()
        .min_by_key(|w| w.active_requests.load(Ordering::Relaxed))
        .unwrap_or(&workers[0])
}
