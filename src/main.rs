use clap::Parser;
use std::{net::SocketAddr, path::PathBuf};
#[derive(Parser)] struct Args {
 #[arg(long,default_value="127.0.0.1:8090",env="DECISION_LAB_BIND")] bind:SocketAddr,
 #[arg(long,default_value="./data/decision-lab.sqlite",env="DECISION_LAB_DB")] db:PathBuf,
 #[arg(long,default_value="./data/artifacts",env="DECISION_LAB_ARTIFACTS")] artifacts:PathBuf,
 #[arg(long,default_value="http://127.0.0.1:8080",env="JEV_LOCAL_URL")] jev_local_url:String,
}
#[tokio::main]async fn main()->Result<(),Box<dyn std::error::Error>>{
 tracing_subscriber::fmt::init();let a=Args::parse();
 let state=decision_lab::initialize(decision_lab::Config{db_path:a.db,artifact_dir:a.artifacts,jev_local_url:a.jev_local_url})?;
 let listener=tokio::net::TcpListener::bind(a.bind).await?;
 println!("decision-lab listening at http://{}",a.bind);
 axum::serve(listener,decision_lab::app(state)).await?;Ok(())
}
