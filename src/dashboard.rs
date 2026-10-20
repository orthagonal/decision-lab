use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Html,
};
use rusqlite::{params, OptionalExtension};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

use crate::AppState;

type Page = (StatusCode, Html<String>);

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn page(status: StatusCode, title: &str, content: &str) -> Page {
    (
        status,
        Html(format!(
            r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{} · Decision Lab</title><style>{}</style></head>
<body><header><a class="brand" href="/">Decision Lab<span>Dataset explorer</span></a></header>
<main>{content}</main><footer>Decision Lab · Dataset explorer</footer></body></html>"#,
            escape(title),
            include_str!("dashboard.css")
        )),
    )
}

fn failure(error: impl std::fmt::Display) -> Page {
    tracing::error!("dashboard: {error}");
    page(StatusCode::INTERNAL_SERVER_ERROR, "Unable to load data",
        "<h1>Unable to load data</h1><p>Please reload the page to try again.</p><a href=\"/\">Back to datasets</a>")
}

pub(crate) async fn index(State(state): State<Arc<AppState>>) -> Page {
    let result = (|| -> rusqlite::Result<String> {
        let db = state.db.lock().unwrap();
        let mut statement = db.prepare("SELECT d.id,d.name,d.created_at,COUNT(s.id) FROM datasets d LEFT JOIN samples s ON s.dataset_id=d.id GROUP BY d.id ORDER BY d.created_at DESC,d.id")?;
        let rows = statement.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, usize>(3)?,
            ))
        })?;
        let mut cards = String::new();
        let mut count = 0;
        let mut entries = 0;
        for row in rows {
            let (id, name, created, total) = row?;
            count += 1;
            entries += total;
            cards.push_str(&format!(r#"<a class="dataset-card" href="/dashboard/datasets/{}">
<span class="eyebrow">DATASET</span><h2>{}</h2><p class="muted">Created <time>{}</time></p>
<div class="card-bottom"><span>{total} {}</span><span class="open">View entries <span aria-hidden="true">→</span></span></div></a>"#,
                escape(&id), escape(&name), escape(&created), if total == 1 { "entry" } else { "entries" }));
        }
        if count == 0 {
            cards.push_str("<section class=\"empty\"><h2>No datasets yet</h2><p>Datasets created through the API appear here. To load the example dataset, run <code>python scripts/demo.py</code> while the service is running.</p></section>");
        }
        Ok(format!(
            r#"<div class="heading"><div><p class="eyebrow">YOUR DATA</p><h1>Datasets</h1><p class="muted">Explore your datasets, one entry at a time.</p></div><a class="button" href="/">Refresh datasets</a></div>
<div class="summary"><span><strong>{count}</strong> datasets</span><span><strong>{entries}</strong> total entries</span></div>
<section class="dataset-grid" aria-label="Datasets">{cards}</section>"#
        ))
    })();
    match result {
        Ok(content) => page(StatusCode::OK, "Datasets", &content),
        Err(error) => failure(error),
    }
}

#[derive(Deserialize)]
pub(crate) struct EntryQuery {
    #[serde(default = "first_entry")]
    entry: usize,
}

fn first_entry() -> usize {
    1
}

fn render_value(raw: &str) -> String {
    let readable = match serde_json::from_str::<Value>(raw) {
        Ok(Value::String(text)) => text,
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| raw.to_owned()),
        Err(_) => raw.to_owned(),
    };
    escape(&readable)
}

fn navigation(id: &str, entry: usize, total: usize) -> String {
    let link = |label: &str, target: Option<usize>| match target {
        Some(target) => format!(
            "<a class=\"button\" href=\"/dashboard/datasets/{}?entry={target}\">{label}</a>",
            escape(id)
        ),
        None => format!("<span class=\"button disabled\" aria-disabled=\"true\">{label}</span>"),
    };
    format!(
        r#"<nav class="pager" aria-label="Entry navigation"><div class="pager-buttons">{}{}</div>
<form method="get"><label for="entry">Entry</label> <input id="entry" name="entry" type="number" min="1" max="{total}" value="{entry}" required> <span>of {total}</span> <button type="submit">Go</button></form>
<div class="pager-buttons">{}{}</div></nav>"#,
        link("First", (entry > 1).then_some(1)),
        link("← Previous", (entry > 1).then_some(entry.saturating_sub(1))),
        link("Next →", (entry < total).then_some(entry.saturating_add(1))),
        link("Last", (entry < total).then_some(total))
    )
}

pub(crate) async fn dataset(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(query): Query<EntryQuery>,
) -> Page {
    let result = (|| -> rusqlite::Result<Option<(String, String)>> {
        let db = state.db.lock().unwrap();
        let dataset: Option<(String, usize)> = db.query_row(
            "SELECT d.name,COUNT(s.id) FROM datasets d LEFT JOIN samples s ON s.dataset_id=d.id WHERE d.id=?1 GROUP BY d.id",
            [&id], |r| Ok((r.get(0)?, r.get(1)?))).optional()?;
        let Some((name, total)) = dataset else {
            return Ok(None);
        };
        let mut content = format!(
            r#"<a class="back" href="/">← All datasets</a><div class="heading"><div><p class="eyebrow">DATASET</p><h1>{}</h1><p class="muted identifier">{}</p></div><span class="count">{total} {}</span></div>"#,
            escape(&name),
            escape(&id),
            if total == 1 { "entry" } else { "entries" }
        );
        if total == 0 {
            content.push_str("<section class=\"empty\"><h2>No entries yet</h2><p>This dataset is empty. Add samples through the dataset API to start exploring.</p></section>");
        } else {
            let entry = query.entry.clamp(1, total);
            let (sample_id, sample_state, questions, expected, split, group): (String, String, String, String, String, Option<String>) = db.query_row(
                "SELECT id,state,questions,expected,split,group_id FROM samples WHERE dataset_id=?1 ORDER BY created_at,id LIMIT 1 OFFSET ?2",
                params![id, entry - 1], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)))?;
            content.push_str(&navigation(&id, entry, total));
            content.push_str(&format!(r#"<article class="entry"><div class="entry-heading"><h2>Entry {entry}</h2><span class="badge">{}</span></div>
<dl class="metadata"><div><dt>Sample ID</dt><dd>{}</dd></div><div><dt>Group</dt><dd>{}</dd></div></dl>
<section class="state"><h3>State</h3><pre>{}</pre></section>
<div class="answer-grid"><section><h3>Questions</h3><pre>{}</pre></section><section><h3>Expected answers</h3><pre>{}</pre></section></div></article>"#,
                escape(&split), escape(&sample_id), escape(group.as_deref().unwrap_or("No group")),
                render_value(&sample_state), render_value(&questions), render_value(&expected)));
        }
        Ok(Some((name, content)))
    })();
    match result {
        Ok(Some((name, content))) => page(StatusCode::OK, &name, &content),
        Ok(None) => page(StatusCode::NOT_FOUND, "Dataset not found", "<h1>Dataset not found</h1><p>This dataset does not exist.</p><a href=\"/\">Back to datasets</a>"),
        Err(error) => failure(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dashboard_browses_entries_and_handles_empty_and_missing_datasets() {
        let directory = tempfile::tempdir().unwrap();
        let state = crate::initialize(crate::Config {
            db_path: directory.path().join("test.sqlite"),
            artifact_dir: directory.path().join("artifacts"),
            jev_local_url: "http://127.0.0.1:8080".into(),
        })
        .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server =
            tokio::spawn(async move { axum::serve(listener, crate::app(state)).await.unwrap() });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();

        let empty = client.get(&base).send().await.unwrap();
        assert_eq!(empty.status(), StatusCode::OK);
        assert!(empty.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/html"));
        assert!(empty.text().await.unwrap().contains("No datasets yet"));

        let created: Value = client
            .post(format!("{base}/datasets"))
            .json(&serde_json::json!({"name": "Door <test>"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let id = created["id"].as_str().unwrap();
        let detail = format!("{base}/dashboard/datasets/{id}");
        assert!(client
            .get(&detail)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
            .contains("No entries yet"));

        let samples = serde_json::json!({"samples": [
            {"state": "first </pre><script>alert(1)</script>", "questions": {"vote": {"type": "choice"}}, "expected": {"vote": "yes"}, "split": "train", "group_id": "scene-1"},
            {"state": "second unique entry", "questions": {"vote": {"type": "choice"}}, "expected": {"vote": "no"}, "split": "test"}
        ]});
        let response = client
            .post(format!("{base}/datasets/{id}/samples"))
            .json(&samples)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let index = client
            .get(format!("{base}/dashboard"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(index.contains("Door &lt;test&gt;"));
        assert!(index.contains(&format!("href=\"/dashboard/datasets/{id}\"")));
        assert!(index.contains("2 entries"));
        let first = client
            .get(&detail)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(first.contains("first &lt;/pre&gt;&lt;script&gt;alert(1)&lt;/script&gt;"));
        assert!(!first.contains("second unique entry"));
        assert!(first.contains("scene-1"));
        assert!(first.contains("Expected answers"));
        assert!(first.contains(&format!("href=\"/dashboard/datasets/{id}?entry=2\">Next")));
        let second = client
            .get(format!("{detail}?entry=2"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(second.contains("second unique entry"));
        assert!(!second.contains("alert(1)"));
        assert!(second.contains("aria-disabled=\"true\">Next"));
        assert!(second.contains("No group"));
        let past_end = client
            .get(format!("{detail}?entry=999"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(past_end.contains("second unique entry"));
        let zero = client
            .get(format!("{detail}?entry=0"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(zero.contains("first &lt;/pre&gt;"));
        let missing = client
            .get(format!("{base}/dashboard/datasets/missing"))
            .send()
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        assert!(missing.text().await.unwrap().contains("Dataset not found"));

        server.abort();
        let _ = server.await;
    }
}
