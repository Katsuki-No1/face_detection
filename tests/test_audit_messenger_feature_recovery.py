import torch

from scripts.audit_messenger_feature_recovery import Accumulator, branches, errors


def test_amplitude_error_is_not_hidden_by_cosine():
    target = torch.tensor([[[1., 2.]]])
    result = errors(2 * target, target, torch.ones(2))
    assert result['cosine_distance'].abs().max() < 1e-6
    assert result['mse'].item() == 2.5
    assert result['scaled_huber'].item() == 1.


def test_patch_weighting_mask_and_tie_tolerance():
    stats = Accumulator()
    stats.add(('teacher','kind','masked'), {'mse': torch.tensor([[4., 2., 8.]])},
              {'output': {'mse': torch.tensor([[1., 3., 0.]])}},
              torch.tensor([[True, True, False]]))
    stats.add(('teacher','kind','masked'), {'mse': torch.tensor([[0.]])},
              {'output': {'mse': torch.tensor([[0.]])}}, torch.tensor([[True]]))
    row = stats.finish()['teacher/kind/masked/output/mse']
    assert row['patches'] == 3
    assert row['raw_error'] == 2
    assert abs(row['branch_error'] - 4/3) < 1e-10
    assert row['improved_fraction'] == row['worsened_fraction'] == row['tied_fraction'] == 1/3


def test_v1_proposal_accounts_for_residual_projection():
    class Block:
        correction_projection = staticmethod(lambda x: 2*x)

        def __call__(self, feature):
            x = feature.flatten(2).transpose(1,2)
            return feature, dict(original_tokens=x, messenger_tokens=x+3,
                                 recovered_tokens=x+1.5)

    raw, out, _ = branches(Block(), torch.ones(1,2,1,1))
    assert torch.equal(out['candidate'], raw+6)
    assert torch.equal(out['output'], raw+1.5)
