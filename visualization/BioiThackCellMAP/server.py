"""
Flask proxy server for NDEx, STRING, and patient data API endpoints.
Avoids browser CORS restrictions when building frontend tools.
"""

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import requests
import os
import patient_loader

app = Flask(__name__, static_folder="static")
CORS(app)

# Load patient data at startup (non-fatal if unavailable)
patient_loader.load()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


NDEX_BASE = "https://www.ndexbio.org/v2"
STRING_BASE = "https://string-db.org/api/json"


@app.route("/api/nest/<network_id>", methods=["GET"])
def get_nest_network(network_id):
    """Fetch a network from NDEx by UUID."""
    url = f"{NDEX_BASE}/network/{network_id}"
    params = dict(request.args)
    try:
        r = requests.get(url, params=params, timeout=60,
                        headers={"Accept": "application/json"})
        return Response(r.content, status=r.status_code, content_type=r.headers.get("Content-Type", "application/json"))
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/string/network", methods=["POST"])
def string_network():
    """
    Query STRING for protein-protein interactions.
    Body (JSON): { identifiers: [...], species: 9606, required_score: 400 }
    """
    body = request.get_json(force=True)
    identifiers = body.get("identifiers", [])
    if not identifiers:
        return jsonify({"error": "identifiers required"}), 400

    payload = {
        "identifiers": "%0d".join(identifiers),
        "species": body.get("species", 9606),
        "required_score": body.get("required_score", 400),
        "network_type": body.get("network_type", "physical"),
        "add_nodes": body.get("add_nodes", 0),
        "caller_identity": "nest_string_tool",
    }
    try:
        r = requests.post(f"{STRING_BASE}/network", data=payload, timeout=30)
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/string/enrichment", methods=["POST"])
def string_enrichment():
    """
    Query STRING for functional enrichment analysis.
    Body (JSON): { identifiers: [...], species: 9606 }
    """
    body = request.get_json(force=True)
    identifiers = body.get("identifiers", [])
    if not identifiers:
        return jsonify({"error": "identifiers required"}), 400

    payload = {
        "identifiers": "%0d".join(identifiers),
        "species": body.get("species", 9606),
        "caller_identity": "nest_string_tool",
    }
    try:
        r = requests.post(f"{STRING_BASE}/enrichment", data=payload, timeout=60)
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/string/image", methods=["POST"])
def string_image():
    """
    Proxy STRING network PNG image to avoid CORS.
    Body (JSON): { identifiers: [...], species: 9606, required_score: 400 }
    """
    body = request.get_json(force=True)
    identifiers = body.get("identifiers", [])
    if not identifiers:
        return jsonify({"error": "identifiers required"}), 400

    payload = {
        "identifiers": "%0d".join(identifiers),
        "species": body.get("species", 9606),
        "required_score": body.get("required_score", 400),
        "network_flavor": body.get("network_flavor", "evidence"),
        "network_type": body.get("network_type", "physical"),
        "caller_identity": "nest_string_tool",
    }
    try:
        r = requests.post("https://string-db.org/api/image/network", data=payload, timeout=30)
        return Response(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "image/png"))
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/string/resolve", methods=["POST"])
def string_resolve():
    """
    Resolve gene identifiers via STRING to get preferred names and descriptions.
    Body (JSON): { identifiers: [...], species: 9606 }
    """
    body = request.get_json(force=True)
    identifiers = body.get("identifiers", [])
    if not identifiers:
        return jsonify({"error": "identifiers required"}), 400

    payload = {
        "identifiers": "%0d".join(identifiers),
        "species": body.get("species", 9606),
        "caller_identity": "nest_string_tool",
    }
    try:
        r = requests.post(f"{STRING_BASE}/resolve", data=payload, timeout=30)
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── Patient data endpoints ────────────────────────────────────────────────

@app.route("/api/patients", methods=["GET"])
def get_patients():
    """Return patient list + metadata for the dropdown."""
    if not patient_loader.is_loaded():
        return jsonify({"error": "patient data not loaded"}), 503
    return jsonify({
        "meta":     patient_loader.get_meta(),
        "patients": patient_loader.get_patient_list(),
    })


@app.route("/api/patient/<path:patient_id>/top-nests", methods=["GET"])
def get_patient_top_nests(patient_id):
    """Return top 10 NeST terms by importance for a patient."""
    if not patient_loader.is_loaded():
        return jsonify({"error": "patient data not loaded"}), 503
    n = int(request.args.get("n", 3))
    result = patient_loader.get_top_nests(patient_id, n=n)
    if result is None:
        return jsonify({"error": f"patient '{patient_id}' not found"}), 404
    preds = patient_loader.get_patient_list()
    pred = next((p["pred"] for p in preds if p["id"] == patient_id), None)
    return jsonify({"patientId": patient_id, "pred": pred, "nests": result})


@app.route("/api/gene/summaries", methods=["POST"])
def gene_summaries():
    """
    Fetch full gene summaries from MyGene.info (avoids STRING truncation).
    Body (JSON): { symbols: ["TP53", "BRCA1", ...] }
    """
    body = request.get_json(force=True)
    symbols = body.get("symbols", [])
    if not symbols:
        return jsonify({"error": "symbols required"}), 400
    try:
        r = requests.post(
            "https://mygene.info/v3/query",
            json={"q": symbols, "scopes": "symbol", "species": "human",
                  "fields": "symbol,name,summary", "size": len(symbols)},
            timeout=10,
        )
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/patient/<path:patient_id>/alterations", methods=["POST"])
def get_patient_alterations(patient_id):
    """
    Return which alteration types (mutation/cnamplification/cndeletion/fusion)
    are present for each requested gene in this patient.
    Body (JSON): { genes: ["TP53", ...] }
    """
    if not patient_loader.is_loaded():
        return jsonify({"error": "patient data not loaded"}), 503
    body  = request.get_json(force=True)
    genes = body.get("genes", [])
    if not genes:
        return jsonify({"error": "genes list required"}), 400
    result = patient_loader.get_gene_alterations(patient_id, genes)
    if result is None:
        return jsonify({"error": f"patient '{patient_id}' not found"}), 404
    return jsonify({"patientId": patient_id, "alterations": result})


@app.route("/api/patient/<path:patient_id>/genes", methods=["POST"])
def get_patient_genes(patient_id):
    """
    Return per-gene importance, z-score, and direction for a patient.
    Body (JSON): { genes: ["TP53", "BRCA1", ...] }
    """
    if not patient_loader.is_loaded():
        return jsonify({"error": "patient data not loaded"}), 503
    body  = request.get_json(force=True)
    genes = body.get("genes", [])
    if not genes:
        return jsonify({"error": "genes list required"}), 400
    result = patient_loader.get_gene_directions(patient_id, genes)
    if result is None:
        return jsonify({"error": f"patient '{patient_id}' not found"}), 404
    return jsonify({"patientId": patient_id, "genes": result})


if __name__ == "__main__":
    print("NeST-STRING proxy server running on http://localhost:5001")
    app.run(port=5001, debug=False)
