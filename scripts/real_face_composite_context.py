"""Query extraction bound to the actual composite pixels for both FOR arms."""
import hashlib
import numpy as np


def extract_composite_context(detector, rgb, composition, *, detector_identity, query_dim):
    rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    if not composition.get('applied') or composition.get('rgb_sha256') != rgb_sha:
        raise ValueError('Composite RGB differs from the recorded composition')
    if not all(detector_identity.get(k) for k in ('checkpoint_sha256', 'config_sha256')):
        raise ValueError('Pinned detector checkpoint/config identity required')
    # Always infer the supplied composite; no clean-image query cache argument.
    rows = detector.predict(rgb)
    seen = set()
    for row in rows:
        index = row.get('face_query_index')
        feature = np.asarray(row.get('face_query_features', []), dtype=float)
        ellipse = np.asarray(row.get('ellipse', []), dtype=float)
        box = np.asarray(row.get('detector_bbox_xyxy', []), dtype=float)
        if (type(index) is not int or index < 0 or index in seen or
                feature.shape != (query_dim,) or not np.isfinite(feature).all() or
                ellipse.shape != (5,) or not np.isfinite(ellipse).all() or
                np.any(ellipse[2:4] <= 0) or box.shape != (4,) or
                not np.isfinite(box).all() or np.any(box[2:] <= box[:2])):
            raise ValueError('Invalid composite detector query/geometry')
        seen.add(index)
    return dict(schema='real-face-composite-context-v1',
                composite_rgb_sha256=rgb_sha,
                composition_mask_sha256=composition['mask_sha256'],
                detector_identity=dict(detector_identity), query_dim=query_dim,
                shared_arms=['for_off', 'for_on'], predictions=rows,
                context_source='fresh_composite_full_image_inference',
                routing_id_semantics='image-local decoder query; not person identity')
