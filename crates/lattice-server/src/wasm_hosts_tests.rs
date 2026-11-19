//! Browser tests for the host bindings: route(), snapshot and JSONL import,
//! the fetch handler, and read-only construction.

use crate::wasm_bindings::LatticeDB;
use js_sys::{Object, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::wasm_bindgen_test;

const CREATE: &str = r#"{"vectors":{"size":3,"distance":"Cosine"}}"#;
const UPSERT: &str =
    r#"{"points":[{"id":1,"vector":[1,0,0],"payload":{"t":"a"}},{"id":2,"vector":[0,1,0]}]}"#;
const SEARCH: &str = r#"{"vector":[1,0,0],"limit":1}"#;

async fn call(db: &LatticeDB, method: &str, path: &str, body: &str) -> (u16, Vec<u8>) {
    let body = if body.is_empty() {
        None
    } else {
        Some(JsValue::from_str(body))
    };
    let promise = db.route_js(method, path, body).expect("valid arguments");
    let out = JsFuture::from(promise).await.expect("route resolves");
    let status = Reflect::get(&out, &"status".into())
        .expect("status")
        .as_f64()
        .expect("number") as u16;
    let body: Uint8Array = Reflect::get(&out, &"body".into())
        .expect("body")
        .unchecked_into();
    (status, body.to_vec())
}

async fn seeded(db: &LatticeDB) {
    assert_eq!(call(db, "PUT", "/collections/c", CREATE).await.0, 200);
    assert_eq!(
        call(db, "put", "/collections/c/points", UPSERT).await.0,
        200
    );
}

fn options(read_only: bool) -> JsValue {
    let o = Object::new();
    Reflect::set(&o, &"readOnly".into(), &JsValue::from_bool(read_only)).expect("set");
    o.into()
}

#[wasm_bindgen_test]
async fn route_drives_the_native_router() {
    let db = LatticeDB::default();
    seeded(&db).await;
    let (status, body) = call(&db, "POST", "/collections/c/points/search", SEARCH).await;
    assert_eq!(status, 200);
    assert!(String::from_utf8_lossy(&body).contains("\"id\":1"));
    assert_eq!(call(&db, "GET", "/nope", "").await.0, 404);
    assert!(db.route_js("", "/ping", None).is_err());
    assert!(db.route_js("GET", "ping", None).is_err());
}

#[wasm_bindgen_test]
async fn snapshot_round_trips_through_an_offset_view() {
    let source = LatticeDB::default();
    seeded(&source).await;
    let snapshot = source.export_snapshot("c").expect("export");
    // Embed the snapshot at an offset inside a larger buffer: a view, not the
    // whole buffer, must be what gets imported.
    let padded = Uint8Array::new_with_length(snapshot.length() + 16);
    padded.set(&snapshot, 8);
    let view = padded.subarray(8, 8 + snapshot.length());

    let mut target = LatticeDB::new(Some(options(true))).expect("options");
    target
        .import_snapshot(view.into(), "copy", "create")
        .expect("import");
    let (status, _) = call(&target, "POST", "/collections/copy/points/search", SEARCH).await;
    assert_eq!(status, 200);
    assert!(target
        .import_snapshot(JsValue::from_str("text"), "x", "create")
        .is_err());
    assert!(target
        .import_snapshot(snapshot.into(), "copy", "bogus")
        .is_err());
}

/// A real `lattice-jsonl` dump of a two-point collection named `site`.
fn site_dump() -> String {
    use lattice_core::{
        export_jsonl, CollectionConfig, CollectionEngine, Distance, ExportOptions, HnswConfig,
        Point, VectorConfig,
    };
    let hnsw = HnswConfig {
        m: 16,
        m0: 32,
        ml: HnswConfig::recommended_ml(16),
        ef: 64,
        ef_construction: 100,
    };
    let config = CollectionConfig::new("site", VectorConfig::new(2, Distance::Cosine), hnsw);
    let mut engine = CollectionEngine::new(config.clone()).expect("engine");
    engine
        .upsert_points(vec![
            Point::new_vector(1, vec![1.0, 0.0]),
            Point::new_vector(2, vec![0.0, 1.0]),
        ])
        .expect("upsert");
    let mut out = Vec::new();
    export_jsonl(&engine, &config, &mut out, &ExportOptions::inline()).expect("export");
    String::from_utf8(out).expect("utf8")
}

#[wasm_bindgen_test]
async fn jsonl_import_names_the_collection() {
    let jsonl = site_dump();
    let mut db = LatticeDB::default();
    db.import_jsonl(JsValue::from_str(&jsonl), "create")
        .expect("import");
    assert_eq!(call(&db, "GET", "/collections/site", "").await.0, 200);
    let bytes = Uint8Array::from(jsonl.as_bytes());
    assert!(db.import_jsonl(bytes.clone().into(), "create").is_err());
    db.import_jsonl(bytes.into(), "replace").expect("replace");
}

#[wasm_bindgen_test]
async fn read_only_refuses_writes_but_serves_reads() {
    let source = LatticeDB::default();
    seeded(&source).await;
    let snapshot = source.export_snapshot("c").expect("export");
    let mut db = LatticeDB::new(Some(options(true))).expect("options");
    db.import_snapshot(snapshot.into(), "c", "create")
        .expect("host import is allowed");
    assert_eq!(
        call(&db, "PUT", "/collections/c/points", UPSERT).await.0,
        403
    );
    assert_eq!(call(&db, "DELETE", "/collections/c", "").await.0, 403);
    let search = call(&db, "POST", "/collections/c/points/search", SEARCH).await;
    assert_eq!(search.0, 200);
    assert!(LatticeDB::new(Some(JsValue::from_str("yes"))).is_err());
}

#[wasm_bindgen_test]
async fn fetch_handler_maps_requests_under_its_prefix() {
    let db = LatticeDB::default();
    seeded(&db).await;
    let o = Object::new();
    Reflect::set(&o, &"prefix".into(), &"/api/lattice".into()).expect("set");
    let api = db.fetch_handler(o.into()).expect("handler");
    assert!(api.handles("https://site.example/api/lattice/collections"));
    assert!(!api.handles("https://site.example/api/latticex"));
    let init = web_sys::RequestInit::new();
    init.set_method("POST");
    init.set_body(&JsValue::from_str(SEARCH));
    let req = web_sys::Request::new_with_str_and_init(
        "https://site.example/api/lattice/collections/c/points/search",
        &init,
    )
    .expect("request");
    let resp: web_sys::Response = JsFuture::from(api.handle_fetch(req))
        .await
        .expect("response")
        .unchecked_into();
    assert_eq!(resp.status(), 200);
    let ctype = resp.headers().get("content-type").expect("header");
    assert_eq!(ctype.as_deref(), Some("application/json"));
}
