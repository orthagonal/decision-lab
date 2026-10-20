use axum::{http::StatusCode, routing::post, Json, Router};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::{net::TcpListener, sync::mpsc, task::JoinHandle};

struct Server {
    url: String,
    task: JoinHandle<()>,
}

impl Server {
    async fn start(router: Router) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Self { url, task }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn gateway_config(backend: jev_local::Backend, upstream: &str) -> jev_local::Config {
    jev_local::Config {
        backend,
        laya_url: upstream.into(),
        jev_url: "http://127.0.0.1:1".into(),
        jev_key: None,
        gateway_key: None,
        benchmark_dataset_dir: None,
        registry: None,
    }
}

struct Lab {
    server: Server,
    client: reqwest::Client,
    _directory: tempfile::TempDir,
}

impl Lab {
    async fn start(gateway: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let state = decision_lab::initialize(decision_lab::Config {
            db_path: directory.path().join("test.sqlite"),
            artifact_dir: directory.path().join("artifacts"),
            jev_local_url: format!("{gateway}/"),
        })
        .unwrap();
        Self {
            server: Server::start(decision_lab::app(state)).await,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(260))
                .build()
                .unwrap(),
            _directory: directory,
        }
    }

    async fn post(&self, path: &str, body: Value) -> reqwest::Response {
        self.client
            .post(format!("{}{path}", self.server.url))
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    async fn get(&self, path: &str) -> Value {
        self.client
            .get(format!("{}{path}", self.server.url))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    async fn dataset(&self, samples: Vec<Value>) -> String {
        let created: Value = self
            .post("/datasets", json!({"name":"jev-local door example"}))
            .await
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();
        let id = created["id"].as_str().unwrap().to_owned();
        self.post(
            &format!("/datasets/{id}/samples"),
            json!({"samples":samples}),
        )
        .await
        .error_for_status()
        .unwrap();
        id
    }

    async fn evaluate(&self, id: &str, model: &str) -> reqwest::Response {
        self.post(
            "/evaluations",
            json!({"dataset_id":id,"model_id":model,"provider":"jev-local","split":"test"}),
        )
        .await
    }
}

fn examples() -> Vec<Value> {
    include_str!("../examples/jev-local-smoke.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[tokio::test]
async fn small_dataset_with_real_jev_local_mock_gateway() {
    let gateway = Server::start(jev_local::app(gateway_config(
        jev_local::Backend::Mock,
        "http://127.0.0.1:1",
    )))
    .await;
    let lab = Lab::start(&gateway.url).await;
    let id = lab.dataset(examples()).await;
    let response = lab.evaluate(&id, "jev-latest").await;
    assert_eq!(response.status(), StatusCode::OK);
    let evaluation: Value = response.json().await.unwrap();
    let result = &evaluation["result"];
    assert_eq!(result["provider"], "jev-local");
    assert_eq!(result["total"], 2);
    // The real gateway's uniform mock always chooses "no" for these two criteria.
    // One correct and one incorrect prediction prove that labels are actually scored.
    assert_eq!(result["correct"], 1);
    assert_eq!(result["accuracy"], 0.5);
    assert_eq!(result["confusion"], json!({"yes":{"no":1},"no":{"no":1}}));
    assert_eq!(result["predictions"].as_array().unwrap().len(), 2);
    let saved = lab.get("/evaluations").await;
    assert_eq!(saved["data"][0]["id"], evaluation["id"]);
    assert_eq!(
        serde_json::from_str::<Value>(saved["data"][0]["result"].as_str().unwrap()).unwrap(),
        *result
    );
    assert!(lab.get("/training-jobs").await["data"]
        .as_array()
        .unwrap()
        .is_empty());
    println!("PASS: dataset → decision-lab → real jev-local mock → persisted evaluation; 1/2 correct (mock transport test)");
}

#[tokio::test]
async fn forwards_unlabeled_samples_through_jev_local_and_scores_provider_predictions() {
    let (sender, mut received) = mpsc::unbounded_channel();
    let fixture = Router::new().route(
        "/v1/systemone",
        post(move |Json(body): Json<Value>| {
            let sender = sender.clone();
            async move {
                sender.send(body.clone()).unwrap();
                let choice = if body["state"] == "Please open the door." {
                    "yes"
                } else {
                    "no"
                };
                Json(json!({"answers":{"vote":{"type":"choice","choice":choice}}}))
            }
        }),
    );
    let upstream = Server::start(fixture).await;
    let gateway = Server::start(jev_local::app(gateway_config(
        jev_local::Backend::Laya,
        &upstream.url,
    )))
    .await;
    let lab = Lab::start(&gateway.url).await;
    let id = lab.dataset(examples()).await;
    let result: Value = lab
        .evaluate(&id, "fixture-model")
        .await
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(result["result"]["accuracy"], 1.0);
    assert_eq!(result["result"]["correct"], 2);
    let mut captured = Vec::new();
    for _ in 0..2 {
        captured.push(
            tokio::time::timeout(Duration::from_secs(5), received.recv())
                .await
                .unwrap()
                .unwrap(),
        );
    }
    for sample in examples() {
        let sent = captured
            .iter()
            .find(|body| body["state"] == sample["state"])
            .unwrap();
        assert_eq!(
            *sent,
            json!({"model":"fixture-model","state":sample["state"],"questions":sample["questions"]})
        );
        assert!(
            sent.get("expected").is_none(),
            "expected labels must never be sent to the model"
        );
    }
    println!("PASS: decision-lab → jev-local Laya forwarding → controlled HTTP fixture; 2/2 correctly scored (no model inference)");
}

#[tokio::test]
async fn provider_failures_and_invalid_predictions_do_not_create_evaluations() {
    for (status, body) in [
        (
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error":"fixture unavailable"}),
        ),
        (StatusCode::OK, json!({"answers":{}})),
        (
            StatusCode::OK,
            json!({"answers":{"vote":{"type":"choice","choice":"unknown"}}}),
        ),
    ] {
        let upstream = Server::start(Router::new().route(
            "/v1/systemone",
            post(move || {
                let body = body.clone();
                async move { (status, Json(body)) }
            }),
        ))
        .await;
        let gateway = Server::start(jev_local::app(gateway_config(
            jev_local::Backend::Laya,
            &upstream.url,
        )))
        .await;
        let lab = Lab::start(&gateway.url).await;
        let id = lab.dataset(examples()).await;
        assert_eq!(
            lab.evaluate(&id, "jev-latest").await.status(),
            StatusCode::BAD_GATEWAY
        );
        assert!(lab.get("/evaluations").await["data"]
            .as_array()
            .unwrap()
            .is_empty());
    }
}

#[tokio::test]
async fn validates_the_whole_split_before_calling_the_gateway() {
    let (sender, mut requests) = mpsc::unbounded_channel();
    let upstream = Server::start(Router::new().route(
        "/v1/systemone",
        post(move || {
            let sender = sender.clone();
            async move {
                sender.send(()).unwrap();
                Json(json!({}))
            }
        }),
    ))
    .await;
    let lab = Lab::start(&upstream.url).await;
    let mut samples = examples();
    samples[1]["expected"]["vote"] = json!("unknown");
    let id = lab.dataset(samples).await;
    assert_eq!(
        lab.evaluate(&id, "jev-latest").await.status(),
        StatusCode::BAD_REQUEST
    );
    assert!(matches!(
        requests.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(lab.get("/evaluations").await["data"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn small_dataset_with_live_jev_local() {
    let gateway = match std::env::var("JEV_LOCAL_URL") {
        Ok(url) if !url.trim().is_empty() => url.trim().to_owned(),
        _ => {
            println!(
                "SKIP: live jev-local test requires JEV_LOCAL_URL in the test process environment"
            );
            return;
        }
    };
    println!("RUN: live jev-local evaluation using JEV_LOCAL_URL");
    let model = std::env::var("JEV_LOCAL_MODEL").unwrap_or_else(|_| "jev-latest".into());
    let lab = Lab::start(&gateway).await;
    let id = lab.dataset(examples()).await;
    let response = lab.evaluate(&id, &model).await;
    let status = response.status();
    let body: Value = response.json().await.unwrap();
    assert_eq!(status, StatusCode::OK, "live provider failed: {body}");
    let result = &body["result"];
    assert_eq!(result["provider"], "jev-local");
    assert_eq!(result["total"], 2);
    let predictions = result["predictions"].as_array().unwrap();
    assert_eq!(predictions.len(), 2);
    let correct = predictions
        .iter()
        .filter(|p| p["expected"] == p["predicted"])
        .count();
    assert_eq!(result["correct"], correct);
    assert_eq!(result["accuracy"], correct as f64 / 2.0);
    println!(
        "Live jev-local result (quality reported, no accuracy threshold): {}",
        serde_json::to_string_pretty(result).unwrap()
    );
}
