use axum::{extract::{Path, State}, http::StatusCode, routing::{get, post}, Json, Router};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::{BTreeMap, HashMap, HashSet}, fs, path::{Path as FsPath, PathBuf}, sync::{Arc, Mutex}};
use uuid::Uuid;

mod dashboard;
mod evaluation;

#[derive(Clone)]
pub struct Config { pub db_path: PathBuf, pub artifact_dir: PathBuf, pub jev_local_url: String }
#[derive(Clone)]
pub struct AppState { pub config: Config, pub db: Arc<Mutex<Connection>>, pub http: reqwest::Client }

type ApiError = (StatusCode, Json<Value>);
fn err(status: StatusCode, message: impl std::fmt::Display) -> ApiError { (status, Json(json!({"error":{"message":message.to_string()}}))) }
fn internal(e: impl std::fmt::Display) -> ApiError { err(StatusCode::INTERNAL_SERVER_ERROR,e) }
fn bad(s: impl std::fmt::Display) -> ApiError { err(StatusCode::BAD_REQUEST,s) }
fn not_found() -> ApiError { err(StatusCode::NOT_FOUND,"not found") }
fn new_id() -> String { Uuid::new_v4().to_string() }
fn now() -> String { chrono::Utc::now().to_rfc3339() }

pub fn initialize(config: Config) -> Result<Arc<AppState>, Box<dyn std::error::Error>> {
    if let Some(parent) = config.db_path.parent() { fs::create_dir_all(parent)?; }
    fs::create_dir_all(&config.artifact_dir)?;
    let conn = Connection::open(&config.db_path)?;
    conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id), state TEXT NOT NULL, questions TEXT NOT NULL, expected TEXT NOT NULL, split TEXT NOT NULL, group_id TEXT, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id), trainer TEXT NOT NULL, status TEXT NOT NULL, model_id TEXT, error TEXT, created_at TEXT NOT NULL, finished_at TEXT);
      CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), dataset_id TEXT NOT NULL, kind TEXT NOT NULL, artifact TEXT NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY, model_id TEXT NOT NULL, dataset_id TEXT NOT NULL, split TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS deployments(id TEXT PRIMARY KEY, model_id TEXT NOT NULL, alias TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL);
    ")?;
    Ok(Arc::new(AppState { config, db: Arc::new(Mutex::new(conn)), http: reqwest::Client::new() }))
}

pub fn app(state: Arc<AppState>) -> Router {
    Router::new()
       .route("/",get(dashboard::index))
       .route("/dashboard",get(dashboard::index))
       .route("/dashboard/datasets/:id",get(dashboard::dataset))
       .route("/health",get(|| async {Json(json!({"status":"ok"}))}))
       .route("/datasets",post(create_dataset).get(list_datasets))
       .route("/datasets/:id/samples",post(add_samples).get(list_samples))
       .route("/datasets/:id/split",post(split_dataset))
       .route("/training-jobs",post(start_job).get(list_jobs))
       .route("/training-jobs/:id",get(get_job))
       .route("/models",get(list_models))
       .route("/evaluations",post(evaluation::evaluate).get(list_evaluations))
       .route("/deployments",post(deploy).get(list_deployments))
       .route("/v1/systemone",post(infer))
       .with_state(state)
}

#[derive(Deserialize)] struct CreateDataset { name: String }
async fn create_dataset(State(s):State<Arc<AppState>>, Json(q):Json<CreateDataset>) -> Result<Json<Value>,ApiError> {
    if q.name.trim().is_empty() {return Err(bad("name required"));}
    let id=new_id(); let dt=now();
    s.db.lock().unwrap().execute("INSERT INTO datasets VALUES (?1,?2,?3)",params![id,q.name,dt]).map_err(internal)?;
    Ok(Json(json!({"id":id,"name":q.name,"created_at":dt})))
}
async fn list_datasets(State(s):State<Arc<AppState>>) -> Result<Json<Value>,ApiError> {
    let db=s.db.lock().unwrap(); let mut st=db.prepare("SELECT id,name,created_at FROM datasets ORDER BY created_at").map_err(internal)?;
    let rows=st.query_map([],|r|Ok(json!({"id":r.get::<_,String>(0)?,"name":r.get::<_,String>(1)?,"created_at":r.get::<_,String>(2)?}))).map_err(internal)?;
    Ok(Json(json!({"data":rows.collect::<Result<Vec<_>,_>>().map_err(internal)?})))
}
#[derive(Deserialize)] struct IncomingSample { state: Value, questions: Value, expected: Value, #[serde(default)] split: Option<String>, #[serde(default)] group_id: Option<String> }
#[derive(Deserialize)] struct AddSamples { samples: Vec<IncomingSample> }
async fn add_samples(State(s):State<Arc<AppState>>,Path(id):Path<String>,Json(q):Json<AddSamples>) -> Result<Json<Value>,ApiError> {
    if q.samples.is_empty() {return Err(bad("samples must not be empty"));}
    let mut db=s.db.lock().unwrap(); let tx=db.transaction().map_err(internal)?;
    let exists:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM datasets WHERE id=?1)",[&id],|r|r.get(0)).map_err(internal)?;
    if !exists {return Err(not_found());}
    for item in &q.samples {
        if !item.questions.is_object() || !item.expected.is_object() {return Err(bad("questions and expected must be objects"));}
        let questions=item.questions.as_object().unwrap(); let expected=item.expected.as_object().unwrap();
        if questions.is_empty() || !questions.keys().all(|k| expected.contains_key(k)) {return Err(bad("every question requires an expected answer"));}
        let split=item.split.as_deref().unwrap_or("unassigned");
        if !["train","validation","test","unassigned"].contains(&split) {return Err(bad("invalid split"));}
        tx.execute("INSERT INTO samples VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",params![new_id(),id,item.state.to_string(),item.questions.to_string(),item.expected.to_string(),split,item.group_id,now()]).map_err(internal)?;
    }
    tx.commit().map_err(internal)?; Ok(Json(json!({"dataset_id":id,"added":q.samples.len()})))
}
#[derive(Clone, Serialize)] pub struct Sample {pub id:String,pub state:Value,pub questions:Value,pub expected:Value,pub split:String,pub group_id:Option<String>}
fn fetch_samples(db:&Connection,id:&str,split:Option<&str>) -> rusqlite::Result<Vec<Sample>> {
    let mut st=db.prepare("SELECT id,state,questions,expected,split,group_id FROM samples WHERE dataset_id=?1 AND (?2 IS NULL OR split=?2) ORDER BY id")?;
    let iter=st.query_map(params![id,split],|r|{
        let state:String=r.get(1)?; let questions:String=r.get(2)?; let expected:String=r.get(3)?;
        Ok(Sample{id:r.get(0)?,state:serde_json::from_str(&state).unwrap_or(Value::Null),questions:serde_json::from_str(&questions).unwrap_or(Value::Null),expected:serde_json::from_str(&expected).unwrap_or(Value::Null),split:r.get(4)?,group_id:r.get(5)?})
    })?;
    iter.collect()
}
async fn list_samples(State(s):State<Arc<AppState>>,Path(id):Path<String>) -> Result<Json<Value>,ApiError> {
    let db=s.db.lock().unwrap(); Ok(Json(json!({"data":fetch_samples(&db,&id,None).map_err(internal)?})))
}
#[derive(Deserialize)] struct SplitConfig { #[serde(default="default_seed")] seed:u64, #[serde(default="default_train")] train:f64, #[serde(default="default_validation")] validation:f64 }
fn default_seed()->u64{42} fn default_train()->f64{0.7} fn default_validation()->f64{0.15}
fn hash_group(v:&str,seed:u64)->u64 {let mut h=14695981039346656037u64^seed; for b in v.as_bytes(){h^=*b as u64;h=h.wrapping_mul(1099511628211);} h}
async fn split_dataset(State(s):State<Arc<AppState>>,Path(id):Path<String>,Json(cfg):Json<SplitConfig>)->Result<Json<Value>,ApiError>{
    if cfg.train<=0.0 || cfg.validation<0.0 || cfg.train+cfg.validation>=1.0 {return Err(bad("require 0<train and 0<=validation and train+validation<1"));}
    let mut db=s.db.lock().unwrap(); let tx=db.transaction().map_err(internal)?;
    let count= {let mut st=tx.prepare("SELECT id,coalesce(group_id,id) FROM samples WHERE dataset_id=?1").map_err(internal)?;
       let rows=st.query_map([&id],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?))).map_err(internal)?;
       let mut n=0;
       for row in rows {let (sample,group)=row.map_err(internal)?; let fraction=(hash_group(&group,cfg.seed)%1_000_000) as f64/1_000_000.0;
         let target=if fraction<cfg.train {"train"}else if fraction<cfg.train+cfg.validation {"validation"}else{"test"};
         tx.execute("UPDATE samples SET split=?1 WHERE id=?2",params![target,sample]).map_err(internal)?; n+=1;
       } n
    };
    tx.commit().map_err(internal)?; Ok(Json(json!({"dataset_id":id,"assigned":count,"seed":cfg.seed})))
}
#[derive(Deserialize)] struct TrainRequest {dataset_id:String,#[serde(default="mock_trainer")]trainer:String}
fn mock_trainer()->String{"mock".to_string()}
async fn start_job(State(s):State<Arc<AppState>>,Json(q):Json<TrainRequest>)->Result<(StatusCode,Json<Value>),ApiError>{
    if q.trainer!="mock" && q.trainer!="laya" {return Err(bad("supported trainers: mock, laya"));}
    let samples={let db=s.db.lock().unwrap();fetch_samples(&db,&q.dataset_id,Some("train")).map_err(internal)?};
    if samples.is_empty(){return Err(bad("dataset needs train samples; call /split first"));}
    let job_id=new_id();let timestamp=now();
    s.db.lock().unwrap().execute("INSERT INTO jobs(id,dataset_id,trainer,status,created_at) VALUES (?1,?2,?3,'queued',?4)",params![job_id,q.dataset_id,q.trainer,timestamp]).map_err(internal)?;
    let s2=s.clone();let jid=job_id.clone();let dataset=q.dataset_id;let trainer=q.trainer;
    let use_mock=trainer=="mock";
    tokio::spawn(async move {
       let result=if use_mock {train_mock(&s2,&jid,&dataset,samples).await} else {train_laya(&s2,&jid,&dataset,samples).await};
       if let Err(e)=result {let _=s2.db.lock().unwrap().execute("UPDATE jobs SET status='failed',error=?1,finished_at=?2 WHERE id=?3",params![e,now(),jid]);}
    });
    Ok((StatusCode::ACCEPTED,Json(json!({"id":job_id,"status":"queued","trainer":trainer}))))
}
#[derive(Serialize,Deserialize)] struct NaiveModel {labels:Vec<String>,prior:HashMap<String,f64>,counts:HashMap<String,HashMap<String,u64>>,totals:HashMap<String,u64>,vocab:HashSet<String>}
fn tokens(text:&str)->Vec<String>{text.to_lowercase().split(|c:char|!c.is_alphanumeric()).filter(|v|!v.is_empty()).map(str::to_string).collect()}
fn text_for(state:&Value)->String {state.as_str().map(str::to_string).unwrap_or_else(||state.to_string())}
fn target(sample:&Sample)->Option<String>{sample.expected.get("vote").and_then(Value::as_str).map(str::to_string)}
fn fit(samples:&[Sample])->Result<NaiveModel,String>{
    let mut totals=HashMap::new();let mut prior:HashMap<String,f64>=HashMap::new();let mut counts:HashMap<String,HashMap<String,u64>>=HashMap::new();let mut vocab=HashSet::new();
    for s in samples {let label=target(s).ok_or("mock trainer requires expected.vote as a string")?;
       *prior.entry(label.clone()).or_default()+=1.0;
       for word in tokens(&text_for(&s.state)){vocab.insert(word.clone());*counts.entry(label.clone()).or_default().entry(word).or_default()+=1;*totals.entry(label.clone()).or_default()+=1;}
    }
    let len=samples.len() as f64;for v in prior.values_mut(){*v/=len;}
    let mut labels:Vec<String>=prior.keys().cloned().collect();labels.sort();
    Ok(NaiveModel{labels,prior,counts,totals,vocab})
}
fn predict(model:&NaiveModel,state:&Value)->BTreeMap<String,f64>{
    let words=tokens(&text_for(state));let mut scores=BTreeMap::new();
    for label in &model.labels {let mut score=model.prior[label].max(1e-12).ln();
       let count=model.counts.get(label);let denom=(*model.totals.get(label).unwrap_or(&0)+model.vocab.len() as u64).max(1) as f64;
       for w in &words {let n=count.and_then(|c|c.get(w)).copied().unwrap_or(0);score+=((n+1) as f64/denom).ln();}
       scores.insert(label.clone(),score);
    }
    let max=scores.values().cloned().fold(f64::NEG_INFINITY,f64::max);
    let z:f64=scores.values().map(|v|(v-max).exp()).sum();for p in scores.values_mut(){*p=(*p-max).exp()/z;}
    scores
}
fn finish_model(s:&Arc<AppState>,job:&str,dataset:&str,kind:&str,artifact:&FsPath)->Result<(),String>{
    let model=new_id();let db=s.db.lock().unwrap();
    db.execute("INSERT INTO models VALUES (?1,?2,?3,?4,?5,?6)",params![model,job,dataset,kind,artifact.to_string_lossy(),now()]).map_err(|e|e.to_string())?;
    db.execute("UPDATE jobs SET status='succeeded',model_id=?1,finished_at=?2 WHERE id=?3",params![model,now(),job]).map_err(|e|e.to_string())?;Ok(())
}
async fn train_mock(s:&Arc<AppState>,job:&str,dataset:&str,samples:Vec<Sample>)->Result<(),String>{
    s.db.lock().unwrap().execute("UPDATE jobs SET status='running' WHERE id=?1",[job]).map_err(|e|e.to_string())?;
    let model=fit(&samples)?;let path=s.config.artifact_dir.join(format!("{job}.json"));
    fs::write(&path,serde_json::to_vec_pretty(&model).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    finish_model(s,job,dataset,"naive-bayes",&path)
}
// Real training is delegated to an installed Laya worker. It must output a checkpoint path.
async fn train_laya(s:&Arc<AppState>,job:&str,dataset:&str,samples:Vec<Sample>)->Result<(),String>{
    s.db.lock().unwrap().execute("UPDATE jobs SET status='running' WHERE id=?1",[job]).map_err(|e|e.to_string())?;
    let command=std::env::var("DECISION_LAB_LAYA_TRAIN_COMMAND").map_err(|_|"set DECISION_LAB_LAYA_TRAIN_COMMAND to a trusted executable trainer path".to_string())?;
    let dataset_path=s.config.artifact_dir.join(format!("{job}-train.jsonl"));
    let lines=samples.iter().map(|s|json!({"state":s.state,"questions":s.questions,"expected":s.expected}).to_string()).collect::<Vec<_>>().join("\n");
    fs::write(&dataset_path,format!("{lines}\n")).map_err(|e|e.to_string())?;
    let output_dir=s.config.artifact_dir.join(format!("{job}-checkpoint"));fs::create_dir_all(&output_dir).map_err(|e|e.to_string())?;
    let result=tokio::process::Command::new(command).arg(&dataset_path).arg(&output_dir).output().await.map_err(|e|e.to_string())?;
    if !result.status.success(){return Err(format!("trainer failed: {}",String::from_utf8_lossy(&result.stderr)));}
    let manifest=output_dir.join("model.json");
    if !manifest.exists(){return Err("trainer did not produce model.json manifest; see workers/laya/README.md".into());}
    finish_model(s,job,dataset,"laya",&manifest)
}
fn query_json_rows(db:&Connection,sql:&str)->Result<Vec<Value>,ApiError>{
    let mut st=db.prepare(sql).map_err(internal)?;
    let col_count=st.column_count();let names=(0..col_count).map(|i|st.column_name(i).unwrap_or("field").to_string()).collect::<Vec<_>>();
    let rows=st.query_map([],|r|{let mut map=serde_json::Map::new();for (i,name) in names.iter().enumerate(){let val:Option<String>=r.get(i)?;map.insert(name.clone(),val.map(Value::String).unwrap_or(Value::Null));}Ok(Value::Object(map))}).map_err(internal)?;
    rows.collect::<Result<Vec<_>,_>>().map_err(internal)
}
async fn list_jobs(State(s):State<Arc<AppState>>)->Result<Json<Value>,ApiError>{Ok(Json(json!({"data":query_json_rows(&s.db.lock().unwrap(),"SELECT * FROM jobs ORDER BY created_at DESC")?})))}
async fn get_job(State(s):State<Arc<AppState>>,Path(id):Path<String>)->Result<Json<Value>,ApiError>{
    let db=s.db.lock().unwrap();let row:Option<(String,String,String,String,Option<String>,Option<String>)>=db.query_row("SELECT id,dataset_id,trainer,status,model_id,error FROM jobs WHERE id=?1",[id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).optional().map_err(internal)?;
    let (id,dataset,trainer,status,model,error)=row.ok_or_else(not_found)?;
    Ok(Json(json!({"id":id,"dataset_id":dataset,"trainer":trainer,"status":status,"model_id":model,"error":error})))
}
async fn list_models(State(s):State<Arc<AppState>>)->Result<Json<Value>,ApiError>{Ok(Json(json!({"data":query_json_rows(&s.db.lock().unwrap(),"SELECT * FROM models ORDER BY created_at DESC")?})))}
fn model_record(db:&Connection,id:&str)->Result<(String,String),ApiError>{db.query_row("SELECT kind,artifact FROM models WHERE id=?1",[id],|r|Ok((r.get(0)?,r.get(1)?))).optional().map_err(internal)?.ok_or_else(not_found)}
async fn list_evaluations(State(s):State<Arc<AppState>>)->Result<Json<Value>,ApiError>{Ok(Json(json!({"data":query_json_rows(&s.db.lock().unwrap(),"SELECT * FROM evaluations ORDER BY created_at DESC")?})))}
#[derive(Deserialize)]struct DeployRequest{model_id:String,alias:String}
async fn deploy(State(s):State<Arc<AppState>>,Json(q):Json<DeployRequest>)->Result<Json<Value>,ApiError>{
    if q.alias.is_empty() || !q.alias.chars().all(|c|c.is_ascii_alphanumeric()||c=='-'||c=='_'){return Err(bad("alias must be alphanumeric, - or _"));}
    let (kind,_)=model_record(&s.db.lock().unwrap(),&q.model_id)?;
    if kind!="naive-bayes"{return Err(bad("this version deploys only mock-trained naive-bayes models; Laya requires runtime integration"));}
    let id=new_id();s.db.lock().unwrap().execute("INSERT INTO deployments(id,model_id,alias,created_at) VALUES (?1,?2,?3,?4) ON CONFLICT(alias) DO UPDATE SET model_id=excluded.model_id,created_at=excluded.created_at",params![id,q.model_id,q.alias,now()]).map_err(internal)?;
    Ok(Json(json!({"id":id,"model_id":q.model_id,"alias":q.alias,"status":"active","endpoint":"/v1/systemone"})))
}
async fn list_deployments(State(s):State<Arc<AppState>>)->Result<Json<Value>,ApiError>{Ok(Json(json!({"data":query_json_rows(&s.db.lock().unwrap(),"SELECT * FROM deployments ORDER BY created_at DESC")?})))}
async fn infer(State(s):State<Arc<AppState>>,Json(v):Json<Value>)->Result<Json<Value>,ApiError>{
    let alias=v.get("model").and_then(Value::as_str).ok_or_else(||bad("model alias required"))?;
    let (kind,artifact)={let db=s.db.lock().unwrap();let model:Option<String>=db.query_row("SELECT model_id FROM deployments WHERE alias=?1",[alias],|r|r.get(0)).optional().map_err(internal)?;
       model_record(&db,&model.ok_or_else(not_found)?)?};
    if kind!="naive-bayes"{return Err(bad("unsupported model kind"));}
    let questions=v.get("questions").and_then(Value::as_object).ok_or_else(||bad("questions object required"))?;
    if questions.len()!=1 || !questions.contains_key("vote") {return Err(bad("v0.1 local inference supports one named choice question: vote"));}
    let state=v.get("state").ok_or_else(||bad("state required"))?;
    let model:NaiveModel=serde_json::from_slice(&fs::read(artifact).map_err(internal)?).map_err(internal)?;
    let probs=predict(&model,state);let choice=probs.iter().max_by(|a,b|a.1.total_cmp(b.1)).map(|(label,_)|label.clone()).unwrap_or_default();
    Ok(Json(json!({"model":alias,"answers":{"vote":{"type":"choice","choice":choice,"probabilities":probs,"confidence":probs.get(&choice)}}})))
}

#[cfg(test)] mod tests {
 use super::*;
 #[test] fn learned_classifier_prefers_examples(){let samples=vec!["open door now", "go through door", "dont open door", "stay away door"].into_iter().zip(["yes","yes","no","no"]).map(|(state,label)|Sample{id:new_id(),state:json!(state),questions:json!({}),expected:json!({"vote":label}),split:"train".into(),group_id:None}).collect::<Vec<_>>();let model=fit(&samples).unwrap();let probs=predict(&model,&json!("go through door"));assert!(probs["yes"]>probs["no"]);}
 #[test] fn grouped_split_hash_stable(){assert_eq!(hash_group("scenario1",42),hash_group("scenario1",42));}
}
