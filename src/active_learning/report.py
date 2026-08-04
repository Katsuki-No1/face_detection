from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Any, Sequence


MODEL_ERROR_TYPES = [
    "missed_detection",
    "false_positive",
    "bbox_localization",
    "low_confidence",
    "temporal_instability",
    "duplicate_detection",
    "correct",
    "other",
]
ATTRIBUTES = [
    "tiny", "small", "profile", "back_head", "occluded", "edge", "overlap", "crowd",
    "dark", "blur", "compression", "mosaic",
]
SELECTION_REASON_LABELS = {
    "model_disagreement": "モデル間の検出結果に差がある",
    "temporal_anomaly": "時系列で顔検出が途切れている",
    "bbox_instability": "bboxの位置・大きさが不安定",
    "low_confidence": "顔検出のconfidenceが判定閾値付近にある",
    "head_evidence": "HEADは検出されたが顔検出がない",
    "dino_novelty": "DINO特徴量が他の候補と異なる",
    "other_instability": "低confidenceまたはbboxが不安定",
    "uniform": "時間方向の均等抽出",
}

BUCKET_LABELS = {
    "model_disagreement": "モデル間の差",
    "temporal_anomaly": "時系列的な途切れ",
    "bbox_instability": "bboxの不安定",
    "low_confidence": "閾値周辺confidence",
    "head_evidence": "headあり",
    "dino_novelty": "DINO特徴量",
    "other_instability": "その他の不安定",
    "uniform": "時間方向の均等抽出",
}

DETAILED_REASON_LABELS = {
    "comparison_detection": "比較対象モデルによる顔検出あり",
    "confidence_drop": "confidenceの低下",
    "disagreement/bbox_variance": "モデル間でbboxの位置・大きさに差がある",
    "disagreement/confidence_variance": "モデル間でconfidenceに差がある",
    "disagreement/count_difference": "モデル間で検出数に差がある",
    "disagreement/iou_variance": "モデル間でbboxの重なり方に差がある",
    "disagreement/unmatched": "一部モデルだけが検出している",
    "disappearance": "顔検出が時系列上で消失した",
    "head_evidence": "headの検出あり",
    "low_confidence": "顔検出のconfidenceが判定閾値付近にある",
    "middle_missing": "前後で検出されているが中間フレームで途切れた",
    "confidence_boundary/closeness": "confidenceがモデルの判定閾値に近い",
    "confidence_boundary/ratio": "confidenceと判定閾値の比率が境界範囲内",
    "no_detection/low_confidence": "confidenceが判定閾値直下のため顔検出なしと判定",
    "no_detection/tracked_missing": "追跡中の顔が検出されなかった",
    "other/bbox_instability": "bboxの位置・大きさが不安定",
    "other/combined": "複数の不安定要因がある",
    "ranked by bbox_instability_score": "bboxの不安定度で優先",
    "ranked by disagreement_score": "モデル間の差で優先",
    "ranked by head_score": "head検出の有無で優先",
    "ranked by low_confidence_score": "低confidenceで優先",
    "ranked by confidence_boundary_score": "判定閾値に近いconfidenceで優先",
    "ranked by novelty_score": "DINO特徴量の希少性で優先",
    "ranked by temporal_anomaly_score": "時系列的な途切れで優先",
    "shape_jump": "bboxの形状が急変した",
    "temporal/confidence_drop": "時系列上でconfidenceが低下した",
    "temporal/disappearance": "時系列上で顔検出が消失した",
    "temporal/middle_missing": "時系列上の中間フレームで検出が途切れた",
    "temporal/shape_jump": "時系列上でbboxの形状が急変した",
}


def selection_reason_label(bucket: str) -> str:
    return SELECTION_REASON_LABELS.get(str(bucket), str(bucket))


def bucket_label(bucket: str) -> str:
    return BUCKET_LABELS.get(str(bucket), str(bucket))


def detailed_reasons_label(value: Any) -> str:
    reasons = [reason for reason in str(value or "").split(";") if reason]
    return "、".join(DETAILED_REASON_LABELS.get(reason, reason) for reason in reasons)


def _safe_column(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in value)


def write_review_template(
    path: Path,
    manifest: Sequence[dict[str, Any]],
    model_names: Sequence[str] = (),
) -> None:
    model_fields = [f"model_{_safe_column(name)}_error_type" for name in model_names]
    fields = ["frame_id", "primary_bucket", "extraction_reason", "annotation_decision", *model_fields, *ATTRIBUTES, "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            bucket = str(row.get("primary_bucket", ""))
            writer.writerow({
                "frame_id": row["frame_id"],
                "primary_bucket": bucket,
                "extraction_reason": row.get("extraction_reason") or selection_reason_label(bucket),
            })


def write_html_report(
    path: Path,
    manifest: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    model_names: Sequence[str] = (),
) -> None:
    """Write an offline review UI whose model controls follow YAML model names."""

    if not model_names:
        model_names = [str(value) for value in summary.get("models", [])]

    def number(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def relative(value: Any) -> str:
        if not value:
            return ""
        return Path(os.path.relpath(Path(str(value)).resolve(), path.parent.resolve())).as_posix()

    cards: list[str] = []
    for row in manifest:
        frame_id = str(row["frame_id"])
        frame_rel = relative(row.get("frame_path"))
        context_rel = relative(row.get("context_clip_path"))
        overlay_rel = relative(row.get("overlay_clip_path"))
        overlay_image_rel = relative(row.get("overlay_image_path"))
        bucket = str(row.get("primary_bucket", ""))
        extraction_reason = str(row.get("extraction_reason") or selection_reason_label(bucket))
        bucket_display = bucket_label(bucket)
        detailed_reasons = detailed_reasons_label(row.get("all_reasons", ""))
        boundary_model = str(row.get("confidence_boundary_model", ""))
        boundary_side = {"below": "閾値直下", "above": "閾値直上", "at": "閾値上"}.get(
            str(row.get("confidence_boundary_side", "")), ""
        )
        boundary_html = ""
        if boundary_model and number(row, "confidence_boundary_score") > 0:
            boundary_html = (
                f'<b>閾値周辺confidence:</b> {html.escape(boundary_model)} — '
                f'confidence {number(row, "confidence_boundary_value"):.4f} ÷ '
                f'閾値 {number(row, "confidence_boundary_threshold"):.4f} = '
                f'{number(row, "confidence_boundary_ratio"):.3f}（{html.escape(boundary_side)}）<br>'
            )
        try:
            counts = json.loads(str(row.get("model_detection_counts_json") or "{}"))
        except json.JSONDecodeError:
            counts = {}
        try:
            predictions = json.loads(str(row.get("model_predictions_json") or "{}"))
        except json.JSONDecodeError:
            predictions = {}
        model_sections: list[str] = []
        for model_name in model_names:
            model_rows = predictions.get(model_name, [])
            prediction_text = ", ".join(
                f"conf={float(item.get('confidence', 0.0)):.3f} "
                f"bbox={item.get('bbox_xyxy_original', [])}"
                for item in model_rows
            ) or "no raw prediction"
            radios = "".join(
                f'<label><input type="radio" name="error-{html.escape(frame_id)}-{html.escape(model_name)}" '
                f'data-model="{html.escape(model_name)}" value="{error_type}"> {error_type}</label>'
                for error_type in MODEL_ERROR_TYPES
            )
            model_sections.append(
                f'<section class="model"><h3>{html.escape(model_name)} '
                f'({int(counts.get(model_name, 0))} analysis-threshold detections)</h3>'
                f'<p class="prediction">{html.escape(prediction_text)}</p>{radios}</section>'
            )
        attributes = "".join(
            f'<label><input type="checkbox" data-attribute="{name}"> {name}</label>'
            for name in ATTRIBUTES
        )
        context_html = (
            f'<video controls preload="metadata" src="{html.escape(context_rel)}"></video>'
            if context_rel else "<p>context clip unavailable</p>"
        )
        overlay_html = (
            f'<video controls preload="metadata" src="{html.escape(overlay_rel)}"></video>'
            if overlay_rel else ""
        )
        cards.append(f"""
<article class="card" data-frame-id="{html.escape(frame_id)}">
  <h2>{html.escape(frame_id)} — {html.escape(str(row['source_video_id']))}</h2>
  <p><b>Source:</b> {html.escape(str(row['source_filename']))} @ {number(row, 'timestamp_sec'):.3f}s,
  frame {html.escape(str(row.get('source_frame_index', '')))}<br>
  <b>抽出根拠:</b> {html.escape(extraction_reason)}<br>
  <b>採用カテゴリ:</b> {html.escape(bucket_display)}（評価値 {number(row, 'selection_score'):.4f}）<br>
  {boundary_html}
  <b>Scores:</b> disagreement={number(row, 'disagreement_score'):.4f}, temporal={number(row, 'temporal_anomaly_score'):.4f}, other={number(row, 'other_instability_score'):.4f}, novelty={number(row, 'novelty_score'):.4f}<br>
  <b>ROI:</b> {html.escape(str(row.get('roi_source', '')))} [{html.escape(str(row.get('roi_bbox', '')))}]<br>
  <b>該当した判定根拠:</b> {html.escape(detailed_reasons)}</p>
  <div class="visuals">
    <figure><a href="{html.escape(frame_rel)}" target="_blank"><img loading="lazy" src="{html.escape(frame_rel)}" alt="original frame"></a><figcaption>Original-resolution dataset frame</figcaption></figure>
    <figure><a href="{html.escape(overlay_image_rel)}" target="_blank"><img loading="lazy" src="{html.escape(overlay_image_rel)}" alt="ROI and model overlays"></a><figcaption>ROI and model bbox overlay</figcaption></figure>
  </div>
  <fieldset class="annotation-decision"><legend>教師データとしての判定</legend>
    <label><input type="radio" name="annotation-{html.escape(frame_id)}" data-annotation-decision value="positive"> 顔あり（positive：正しいbboxを付与）</label>
    <label><input type="radio" name="annotation-{html.escape(frame_id)}" data-annotation-decision value="negative"> 顔なし（negative：bboxなし）</label>
    <label><input type="radio" name="annotation-{html.escape(frame_id)}" data-annotation-decision value="needs_review"> 要再確認</label>
  </fieldset>
  <details><summary>Context ±2 seconds / model overlay video</summary>{context_html}{overlay_html}</details>
  <fieldset><legend>Model errors</legend>{''.join(model_sections)}</fieldset>
  <fieldset><legend>Attributes</legend>{attributes}</fieldset>
  <textarea placeholder="notes"></textarea>
</article>""")

    summary_json = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Active Learning Frame Review</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#101318;color:#e8edf2}}header{{position:sticky;top:0;background:#18202bee;padding:14px 3%;backdrop-filter:blur(8px);z-index:2}}main{{width:min(1600px,94%);margin:auto}}.card{{background:#1b222c;margin:24px 0;padding:22px;border-radius:12px}}.visuals{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}figure{{margin:0}}video,img{{width:100%;max-height:75vh;object-fit:contain;background:#000}}fieldset,.model{{margin:12px 0;border:1px solid #4b596a;padding:12px}}label{{display:inline-block;margin:6px 12px 6px 0}}textarea{{box-sizing:border-box;width:100%;min-height:72px}}button{{padding:9px 14px;margin-right:8px}}pre,.prediction{{white-space:pre-wrap;overflow-wrap:anywhere}}figcaption{{color:#aeb8c4}}
</style></head><body><header><button onclick="downloadReview('json')">Download JSON</button><button onclick="downloadReview('csv')">Download CSV</button> <span id="saved"></span></header>
<main><details><summary>Run summary</summary><pre>{summary_json}</pre></details>{''.join(cards)}</main>
<script>
const models={json.dumps(list(model_names), ensure_ascii=False)};
const attributes={json.dumps(ATTRIBUTES)};
const key='active-learning-frame-review:'+location.pathname;
function collect(){{return [...document.querySelectorAll('.card')].map(card=>{{const errors={{}};for(const model of models){{const selected=[...card.querySelectorAll('input[type=radio][data-model]')].find(x=>x.dataset.model===model&&x.checked);errors[model]=selected?selected.value:''}}const decision=card.querySelector('input[data-annotation-decision]:checked');const attrs=[...card.querySelectorAll('input[type=checkbox]:checked')].map(x=>x.dataset.attribute);return {{frame_id:card.dataset.frameId,annotation_decision:decision?decision.value:'',model_errors:errors,attributes:attrs,notes:card.querySelector('textarea').value}}}})}}
function persist(){{localStorage.setItem(key,JSON.stringify(collect()));document.getElementById('saved').textContent='saved '+new Date().toLocaleTimeString()}}
document.addEventListener('change',persist);document.addEventListener('input',persist);
try{{const rows=JSON.parse(localStorage.getItem(key)||'[]');for(const row of rows){{const card=document.querySelector(`[data-frame-id="${{CSS.escape(row.frame_id)}}"]`);if(!card)continue;const decision=card.querySelector(`input[data-annotation-decision][value="${{CSS.escape(row.annotation_decision||'')}}"]`);if(decision)decision.checked=true;for(const [model,value] of Object.entries(row.model_errors||{{}})){{const radio=[...card.querySelectorAll('input[type=radio][data-model]')].find(x=>x.dataset.model===model&&x.value===value);if(radio)radio.checked=true}}for(const value of row.attributes||[]){{const box=card.querySelector(`[data-attribute="${{CSS.escape(value)}}"]`);if(box)box.checked=true}}card.querySelector('textarea').value=row.notes||''}}}}catch(error){{console.warn(error)}}
function downloadReview(format){{const rows=collect();let body,type,name;if(format==='json'){{body=JSON.stringify(rows,null,2);type='application/json';name='active_learning_review.json'}}else{{const head=['frame_id','annotation_decision',...models.map(x=>'model_'+x.replace(/[^A-Za-z0-9_-]/g,'_')+'_error_type'),...attributes,'notes'];const esc=x=>'"'+String(x??'').replaceAll('"','""')+'"';body='\uFEFF'+[head,...rows.map(r=>[r.frame_id,r.annotation_decision,...models.map(m=>r.model_errors[m]||''),...attributes.map(a=>r.attributes.includes(a)?'1':'0'),r.notes])].map(r=>r.map(esc).join(',')).join('\\r\\n');type='text/csv;charset=utf-8';name='active_learning_review.csv'}}const a=document.createElement('a');const url=URL.createObjectURL(new Blob([body],{{type}}));a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}}
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
