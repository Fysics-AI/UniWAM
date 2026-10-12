"""Behavioral checks for the full model with tiny real DiT blocks on CPU."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/src'))

import pytest
import torch
from torch import nn
from fastwam.models.wan22.uniwam import UniWAM
from fastwam.models.wan22.uniwam_action_dit import UniWAMActionDiT
from fastwam.models.wan22.uniwam_initialization import initialize_action_from_video
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.mot import MoT
from fastwam.datasets.robot_geometry import (rotation_to_6d, rotation_from_6d, poses_in_camera,
    manipulation_targets, navigation_targets, camera_canvas)
from fastwam.datasets.robot_windows import IndependentWindowSampler


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4
    def encode(self, video, **kwargs):
        value = video[:, :, ::4, ::8, ::8]
        return torch.cat((value, value[:, :1]), 1)


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    common = dict(num_heads=2, attn_head_dim=12, num_layers=2, text_dim=16, freq_dim=8, eps=1e-6)
    video = WanVideoDiT(hidden_dim=24, ffn_dim=48, in_dim=4, out_dim=4, has_image_input=False,
        patch_size=(1,2,2), seperated_timestep=True, fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode='first_frame_causal', **common)
    action = UniWAMActionDiT(hidden_dim=16, ffn_dim=32, manip_action_dim=26, nav_action_dim=9,
        manip_horizon=4, nav_horizon=4, type_token_count=2, **common)
    initialize_action_from_video(action, video)
    return UniWAM(video_expert=video, action_expert=action, mot=MoT({'video': video, 'action': action}, False),
        vae=TinyVAE(), text_dim=16, proprio_dim=20, prefix_min=1, prefix_max=2, device='cpu')


def sample(branches=('manip', 'nav')):
    n = len(branches)
    result = dict(context=torch.randn(n,3,16), context_mask=torch.ones(n,3,dtype=torch.bool),
                  proprio=torch.randn(n,1,20))
    for branch, dim in (('manip',26), ('nav',9)):
        count = branches.count(branch)
        result[f'{branch}_branch_valid'] = torch.tensor([x == branch for x in branches])
        if count:
            result[f'{branch}_video'] = torch.rand(count,3,9,32,32)*2-1
            result[f'{branch}_action'] = torch.randn(count,4,dim)
            result[f'{branch}_action_feature_mask'] = torch.ones(count,4,dim,dtype=torch.bool)
            result[f'{branch}_prefix_lengths'] = torch.ones(count,dtype=torch.long)
    return result


@pytest.mark.parametrize('branches', [('manip',), ('nav',), ('manip','nav','nav')])
def test_training_forward_backward(model, branches):
    loss, metrics = model(sample(branches))
    assert torch.isfinite(loss) and len(metrics) == 5
    loss.backward()
    for parameter in model.mot.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.action_expert.blocks[0].self_attn.q.weight.grad.abs().sum() > 0


def test_no_future_or_cross_row_leakage(model):
    model.eval()
    data = sample(('manip','nav'))
    inputs = model.build_inputs(data)
    video = inputs['packed_input_latents']
    actions = [inputs['manip_action'], inputs['nav_action']]
    def forward(v, aa):
        vp = model.video_expert.pre_dit(v, torch.ones(2)*500, inputs['packed_context'], inputs['packed_context_mask'], fuse_vae_embedding_in_latents=True)
        vp = model._prepend_view_type_token(vp, inputs['stream_kind'])
        ap = model.action_expert.pre_dit_packed(manip_action=aa[0], nav_action=aa[1],
            manip_timestep=torch.ones(1)*500, nav_timestep=torch.ones(1)*500,
            manip_context=inputs['packed_context'][:1], nav_context=inputs['packed_context'][1:],
            manip_context_mask=inputs['packed_context_mask'][:1], nav_context_mask=inputs['packed_context_mask'][1:])
        return model._packed_mot_forward(vp, ap)
    expected = forward(video, actions)
    future = video.clone(); future[:,:,1:] += 20
    actual = forward(future, actions)
    torch.testing.assert_close(actual['action'], expected['action'])
    torch.testing.assert_close(actual['video'][:,:6], expected['video'][:,:6])
    changed = [x.clone() for x in actions]; changed[0][:,2:] += 30
    actual = forward(video, changed)
    torch.testing.assert_close(actual['action'][0,:4], expected['action'][0,:4])
    torch.testing.assert_close(actual['action'][1], expected['action'][1])
    row = video.clone(); row[0] += 50
    actual = forward(row, actions)
    torch.testing.assert_close(actual['action'][1], expected['action'][1])
    assert not torch.allclose(actual['action'][0], expected['action'][0])


@pytest.mark.parametrize('branches', [('manip',), ('nav',), ('manip','nav')])
def test_inference_clamps_prefix_and_masks(model, branches):
    prefixes, masks, images = {}, {}, {}
    for branch in branches:
        dim = 26 if branch == 'manip' else 9
        prefixes[branch] = torch.randn(2,dim)
        masks[branch] = torch.ones(4,dim,dtype=torch.bool)
        masks[branch][:,-1] = False
        images[f'{branch}_input_image'] = torch.zeros(3,32,32,dtype=torch.uint8)
    result = model.infer_action(action_horizon=4, context=torch.randn(3,16), context_mask=torch.ones(3,dtype=torch.bool),
        proprio=torch.zeros(20), num_inference_steps=3, seed=1, committed_prefix=prefixes, action_feature_mask=masks, **images)
    for branch in branches:
        out = result[f'{branch}_action']
        assert out.shape == (4, masks[branch].shape[-1]) and torch.isfinite(out).all()
        torch.testing.assert_close(out[:2], torch.where(masks[branch][:2], prefixes[branch], 0))
        assert not out[:,-1].any()


def test_masked_group_loss_ignores_absent_arm_and_prefix(model):
    pred = torch.zeros(2,4,20,requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred,dtype=torch.bool); mask[:,:,10:] = False
    target.data[:,:,10:] = 100
    target.data[:,0,:] = 100
    pad = torch.zeros(2,4,dtype=torch.bool); pad[:,0] = True
    total,count = model._masked_action_sum_count(pred,target,mask,pad,torch.ones(2,dtype=torch.bool),torch.ones(2)*500,pred.sum()*0)
    assert total == 0 and count == 2
    total.backward()
    assert not pred.grad.any()


def test_geometry_and_single_arm_auxiliary_masks():
    rot = torch.eye(3).expand(4,2,3,3).clone()
    rot[:,0] = torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
    torch.testing.assert_close(rotation_from_6d(rotation_to_6d(rot)), rot)
    camera = torch.eye(4); camera[0,3] = 2
    pos, transformed = poses_in_camera(torch.zeros(4,2,3), rot, camera, torch.eye(4).expand(2,4,4))
    assert (pos[:,:,0] == 2).all()
    uv = torch.ones(4,2,2) * .5
    quality = torch.tensor([[1,1],[2,1],[0,1],[1,1]])
    uv[1:3,0] = float('nan')
    targets, mask = manipulation_targets(pos,transformed,torch.zeros(4,2),[True,False],uv,quality)
    assert targets.shape == (4,26) and torch.isfinite(targets).all()
    assert not mask[:,10:20].any() and not mask[:,23:].any()
    assert not mask[1:3,20:22].any() and mask[1,22] and not mask[2,22]
    assert targets[0,22] == 1 and targets[1,22] == 0
    targets, mask = navigation_targets(torch.tensor([[1.,2.,3.1415927]]),torch.tensor([1.,1.,1.5707963]),torch.zeros(1,4),torch.zeros(1,2),torch.ones(1,6,dtype=torch.bool))
    torch.testing.assert_close(targets[0,:3],torch.tensor([1.,0.,1.5707963]),atol=1e-6,rtol=1e-5)
    assert camera_canvas(torch.zeros(9,3,20,20,dtype=torch.uint8)).shape == (3,9,320,384)


def test_source_mass_and_distributed_separation():
    rows = [{'source': s,'stream': 'nav','episode': str(i),'start': 0,'active': True}
            for s,count in [('large',100),('small',10)] for i in range(count)]
    sampler = IndependentWindowSampler(rows,2,5,world_size=2)
    assert sampler.probability[:100].sum() == pytest.approx(.5)
    other = IndependentWindowSampler(rows,2,5,rank=1,world_size=2)
    for a,b in zip(sampler,other): assert len(set(a+b)) == 4
    rows = [{'source':'x','stream':'nav','episode':'e','start':i,'active':True} for i in range(10)]
    with pytest.raises(ValueError,match='Cannot draw'):
        next(iter(IndependentWindowSampler(rows,2,1,min_start_separation=33)))


def test_training_prefix_is_clean_and_has_no_loss(model):
    data = sample(('manip',))
    data['manip_prefix_lengths'] = torch.tensor([2])
    data['manip_action_feature_mask'][:,:,10:20] = False
    observed, outputs = [], []
    handle = model.action_expert.manip_action_encoder.register_forward_pre_hook(lambda module,args: observed.append(args[0].detach().clone()))
    def record_output(module, args, value):
        value.retain_grad(); outputs.append(value)
    output_handle = model.action_expert.manip_head.register_forward_hook(record_output)
    loss,_ = model(data)
    loss.backward()
    clean = torch.where(data['manip_action_feature_mask'], data['manip_action'], 0)
    torch.testing.assert_close(observed[0][:,:2], clean[:,:2])
    assert not outputs[0].grad[:,:2].any()
    assert not outputs[0].grad[:,:,10:20].any()
    assert outputs[0].grad[:,2:,:10].abs().sum() > 0
    handle.remove(); output_handle.remove()


def test_relocated_data_contract_and_entrypoint(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import numpy as np
    from fastwam.datasets.robot_windows import RobotWindows, collate_robot_windows, relative_file
    data_root = tmp_path / 'robot data with spaces'; data_root.mkdir()
    stats = {name: {'offset':[0.] * n,'scale':[2.] * n} for name,n in [('nav',9),('manip',26),('proprio',20)]}
    (data_root/'normalization.json').write_text(json.dumps(stats))
    torch.save({'context':torch.zeros(128,4096),'mask':torch.ones(128,dtype=torch.bool)},data_root/'text.pt')
    rows=[]
    for branch,dim in [('manip',26),('nav',9)]:
        mask=np.ones((32,dim),dtype=bool)
        action=np.ones((32,dim),dtype=np.float32)
        if branch == 'manip':
            mask[:,10:20]=False; action[:,10:20]=np.nan
        np.savez_compressed(data_root/f'{branch}.npz',video=np.zeros((3,9,320,384),dtype=np.uint8),
            action=action,feature_mask=mask,action_is_pad=np.zeros(32,dtype=bool),image_is_pad=np.zeros(9,dtype=bool),
            proprio=np.zeros(20,dtype=np.float32),proprio_mask=np.ones(20,dtype=bool))
        rows.append(dict(source='task',episode=branch,start=0,stream=branch,active=True,path=f'{branch}.npz',text='text.pt'))
    (data_root/'windows.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    monkeypatch.chdir(tmp_path)
    dataset=RobotWindows(data_root)
    batch=collate_robot_windows([dataset[0],dataset[1]])
    assert batch['manip_action'].shape == (1,32,26)
    assert batch['nav_action'].shape == (1,32,9)
    assert batch['context'].shape == (2,128,4096)
    assert torch.isfinite(batch['manip_action']).all()
    assert not batch['manip_action'][:,:,10:20].any()
    assert (batch['nav_action'] == .5).all()
    torch.testing.assert_close(dataset.normalizer.decode('nav',batch['nav_action']),torch.ones_like(batch['nav_action']))
    with pytest.raises(ValueError): relative_file(data_root,'../escape.npz')
    script=Path(__file__).resolve().parents[1]/'training/scripts/train_robot.py'
    result=subprocess.run([sys.executable,str(script),'--data-root',str(data_root),'--data-only'],cwd=tmp_path,text=True,capture_output=True,timeout=40)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])['nav_action'] == [1,32,9]
