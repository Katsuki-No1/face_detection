import pytest

from scripts.audit_messenger_natural_occlusion import discrimination, effect, keypoint_metrics


def test_tied_scores_are_chance_and_ap_equals_prevalence():
    m=discrimination([True,False,False,False],[.2]*4)
    assert m['auroc']==.5
    assert m['average_precision']==.25
    assert m['thresholds']['0.3']['recall']==0


def test_roc_and_ap_handle_ties_at_threshold_groups():
    m=discrimination([False,True,False,True],[0,.5,.5,1])
    assert m['auroc']==.875
    assert m['average_precision']==pytest.approx(5/6)


def test_effect_counts_presence_misses_and_localization_failures():
    rows=[dict(slot=0,state=1,raw=dict(nme=.03,presence=.2),recovered=dict(nme=.04,presence=.9)),
          dict(slot=1,state=1,raw=dict(nme=.03,presence=.9),recovered=dict(nme=.2,presence=.9))]
    e=effect(rows,'recovered')
    assert e['rescued']==e['lost']==1
    assert e['coordinate_worsened']==2
    assert keypoint_metrics(rows,'recovered',1)['semantic_macro_f1_nme_0_10']==.5
