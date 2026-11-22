import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

spec=importlib.util.spec_from_file_location('prior_evaluation',Path(__file__).resolve().parents[1]/'scripts/evaluate_messenger_geometry_prior.py')
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_inference_keeps_unmatched_predictions_and_empty_images(tmp_path):
    Image.fromarray(np.zeros((100,120,3),dtype=np.uint8)).save(tmp_path/'image.png')
    class Model(torch.nn.Module):
        def forward(self,images,*,geometry,support):
            n=len(images)
            assert geometry.shape==(n,3) and support.shape==(n,16,16)
            assert support.flatten(1).any(1).all()
            return dict(decoded_coordinates=torch.full((n,5,2),.5),presence_logits=torch.zeros(n,5),
                state_probabilities=torch.full((n,5,3),1/3))
    coco=dict(images=[dict(id=1,file_name='image.png'),dict(id=2,file_name='unused.png')],annotations=[])
    detections={1:[dict(ellipse=[40,50,20,10,0],bbox_xyxy=[20,40,60,60],score=.8),
                   dict(ellipse=[80,50,20,10,0],bbox_xyxy=[60,40,100,60],score=.7)],2:[]}
    raw=dict(data=dict(root=str(tmp_path),input_size=32,augmentation=dict(scale_factor=1.25),mean=[.5]*3,std=[.5]*3))
    result=evaluation.infer(Model(),coco,detections,raw,device=torch.device('cpu'),batch_size=1)
    assert len(result[1]['poses'])==len(result[1]['faces'])==2
    assert result[2]==dict(heads=[],faces=[],poses=[])
    assert result[1]['poses'][0]['score']==.8
    assert result[1]['poses'][1]['keypoints'][0][0]>result[1]['poses'][0]['keypoints'][0][0]


def test_model_loader_rejects_diagnostic_or_foreign_checkpoint(tmp_path):
    path=tmp_path/'checkpoint.pt'
    for mode in ('launcher-check','production'):
        torch.save(dict(schema='geometry-prior-runtime-v1',contract=dict(experiment=dict(
            identity={'wrong':True},architecture='control',mode=mode))),path)
        with pytest.raises(ValueError,match='authenticated production'):
            evaluation.load_model(dict(architecture='control'),{},path,torch.device('cpu'))


def test_cuda_context_precedes_memory_statistics(monkeypatch):
    calls=[]
    monkeypatch.setattr(torch.cuda,'set_device',lambda index:calls.append(('set',index)))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda index:calls.append(('reset',index)))
    assert evaluation.prepare_cuda()==torch.device('cuda:0')
    assert calls==[('set',0),('reset',0)]
