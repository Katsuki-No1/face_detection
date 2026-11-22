"""Shared scalar-only epoch validation for matched FOR off/on experiments."""
import torch
from torch.utils.data import DataLoader
from keypoint_comparison1.landmark_detection.dataset import collate_landmark_batch
from keypoint_comparison1.landmark_detection.evaluation import predictions_to_original
from keypoint_comparison1.landmark_detection.face_overlap import face_context_kwargs
from keypoint_comparison1.landmark_detection.geometry_prior_validation import ConditionalSemanticCounters, CONTRACT


@torch.inference_mode()
def validate_for(model, dataset, *, device, batch_size, num_workers=0):
    report = dataset.detector_crop_report
    if (dataset.training or dataset.face_overlap_query_dim != 224 or not report
            or report.get('matching_contract') != CONTRACT
            or report.get('normalization_bbox_source') != 'GT Head'):
        raise ValueError('FOR validation requires shared unaugmented predicted FACE/GT Head assignments')
    counters = ConditionalSemanticCounters()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate_landmark_batch)
    model.eval()
    for images, targets in loader:
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            output = model(images.to(device), **face_context_kwargs(model, targets, device))
        inverse = torch.stack([t['inverse_affine'] for t in targets])
        points = predictions_to_original(output['decoded_coordinates'], inverse, input_size=dataset.input_size)
        if 'presence_logits' not in output:
            raise ValueError('FOR Query presence branch missing')
        counters.add(points, output['presence_logits'].sigmoid().float().cpu().numpy(), targets)
    if counters.instances != len(dataset) or not counters.instances:
        raise ValueError('Incomplete or empty FOR validation')
    return dict(counters.compute(), detector_crop_report=report)
