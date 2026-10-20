use axum::{extract::State, http::StatusCode, Json};
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, sync::Arc, time::Duration};

use crate::{
    bad, err, fetch_samples, internal, model_record, new_id, now, predict, target, ApiError,
    AppState, NaiveModel, Sample,
};

#[derive(Deserialize)]
pub(crate) struct EvalRequest {
    model_id: String,
    dataset_id: String,
    #[serde(default = "test_split")]
    split: String,
    #[serde(default = "local_provider")]
    provider: String,
}

fn test_split() -> String {
    "test".into()
}
fn local_provider() -> String {
    "local".into()
}

fn validate_remote_sample(sample: &Sample) -> Result<(), ApiError> {
    let questions = sample
        .questions
        .as_object()
        .ok_or_else(|| bad("questions must be an object"))?;
    let question = &sample.questions["vote"];
    let criteria = question["criteria"].as_object();
    let truth = target(sample).ok_or_else(|| bad("expected.vote must be a string"))?;
    if questions.len() != 1
        || question["type"] != "choice"
        || !question["instructions"].is_string()
        || !criteria.is_some_and(|options| options.len() >= 2 && options.contains_key(&truth))
        || sample.state.is_null()
    {
        return Err(bad("jev-local evaluation requires a non-null state and one vote choice question with instructions, at least two criteria, and expected.vote matching a criterion"));
    }
    Ok(())
}

async fn remote_prediction(
    state: &AppState,
    model: &str,
    sample: &Sample,
) -> Result<String, ApiError> {
    let response = state
        .http
        .post(format!(
            "{}/v1/systemone",
            state.config.jev_local_url.trim_end_matches('/')
        ))
        .timeout(Duration::from_secs(120))
        .json(&json!({"model": model, "state": sample.state, "questions": sample.questions}))
        .send()
        .await
        .map_err(|error| {
            err(
                StatusCode::BAD_GATEWAY,
                format!("jev-local unavailable: {error}"),
            )
        })?;
    if !response.status().is_success() {
        return Err(err(
            StatusCode::BAD_GATEWAY,
            format!("jev-local returned HTTP {}", response.status()),
        ));
    }
    let body: Value = response
        .json()
        .await
        .map_err(|_| err(StatusCode::BAD_GATEWAY, "jev-local returned invalid JSON"))?;
    let answer = &body["answers"]["vote"];
    let predicted = answer["choice"]
        .as_str()
        .filter(|label| {
            answer["type"] == "choice" && sample.questions["vote"]["criteria"].get(*label).is_some()
        })
        .ok_or_else(|| {
            err(
                StatusCode::BAD_GATEWAY,
                "jev-local returned an invalid vote choice",
            )
        })?;
    Ok(predicted.to_owned())
}

pub(crate) async fn evaluate(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EvalRequest>,
) -> Result<Json<Value>, ApiError> {
    if !["train", "validation", "test"].contains(&request.split.as_str()) {
        return Err(bad("split must be train, validation, or test"));
    }
    if !["local", "jev-local"].contains(&request.provider.as_str()) {
        return Err(bad("supported evaluation providers: local, jev-local"));
    }
    if request.model_id.trim().is_empty() {
        return Err(bad("model_id required"));
    }
    let (kind, model) = if request.provider == "local" {
        let (kind, artifact) = model_record(&state.db.lock().unwrap(), &request.model_id)?;
        if kind != "naive-bayes" {
            return Err(bad("Laya evaluation requires a registered inference runtime; native evaluation not yet implemented"));
        }
        let model: NaiveModel =
            serde_json::from_slice(&fs::read(artifact).map_err(internal)?).map_err(internal)?;
        (kind, Some(model))
    } else {
        ("jev-local".to_owned(), None)
    };
    let samples = fetch_samples(
        &state.db.lock().unwrap(),
        &request.dataset_id,
        Some(&request.split),
    )
    .map_err(internal)?;
    if samples.is_empty() {
        return Err(bad("no samples in requested evaluation split"));
    }
    // Validate the whole split before contacting a provider.
    for sample in &samples {
        target(sample).ok_or_else(|| bad("expected.vote must be a string"))?;
        if request.provider == "jev-local" {
            validate_remote_sample(sample)?;
        }
    }
    let mut correct = 0;
    let mut confusion: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    let mut predictions = Vec::with_capacity(samples.len());
    for sample in &samples {
        let truth = target(sample).unwrap();
        let predicted = match &model {
            Some(model) => predict(model, &sample.state)
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(label, _)| label)
                .unwrap_or_default(),
            None => remote_prediction(&state, &request.model_id, sample).await?,
        };
        if predicted == truth {
            correct += 1;
        }
        *confusion
            .entry(truth.clone())
            .or_default()
            .entry(predicted.clone())
            .or_default() += 1;
        predictions
            .push(json!({"sample_id": sample.id, "expected": truth, "predicted": predicted}));
    }
    let result = json!({"accuracy": correct as f64 / samples.len() as f64, "correct": correct,
        "total": samples.len(), "confusion": confusion, "model_kind": kind,
        "provider": request.provider, "predictions": predictions});
    let id = new_id();
    state
        .db
        .lock()
        .unwrap()
        .execute(
            "INSERT INTO evaluations VALUES (?1,?2,?3,?4,?5,?6)",
            params![
                id,
                request.model_id,
                request.dataset_id,
                request.split,
                result.to_string(),
                now()
            ],
        )
        .map_err(internal)?;
    Ok(Json(json!({"id": id, "result": result})))
}
