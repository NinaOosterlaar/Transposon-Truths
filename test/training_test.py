import torch
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.training.training import add_noise

torch.manual_seed(0)


def test_no_noise():
    y = torch.tensor([[0., 1., 2.],
                      [3., 0., 4.]])

    y_noisy, mask = add_noise(y, denoise_percent=0.0)

    assert torch.equal(y, y_noisy), "Tensor should be unchanged when denoise_percent = 0"
    assert mask.dtype == torch.bool, "Mask should be boolean"
    assert mask.shape == y.shape, "Mask should have same shape as input"
    assert mask.sum().item() == 0, "Mask should be all-False when denoise_percent = 0"


def test_only_nonzeros_are_zeroed_and_mask_matches():
    y = torch.tensor([[0., 1., 2.],
                      [3., 0., 4.]])

    y_noisy, mask = add_noise(y, denoise_percent=0.5)

    # Original zeros must stay zero
    assert torch.all(y[y == 0] == y_noisy[y == 0]), "Original zeros should never change"

    # Masked positions must be positions that were originally non-zero
    assert torch.all(y[mask] != 0), "Mask can only be True at originally non-zero positions"

    # Wherever mask is True, y_noisy must be zero
    assert torch.all(y_noisy[mask] == 0), "Masked positions must be set to zero"

    # Wherever mask is False, values must be unchanged (important: ensures we didn't touch other entries)
    assert torch.all(y_noisy[~mask] == y[~mask]), "Unmasked positions must remain unchanged"


def test_correct_number_zeroed_per_sequence():
    y = torch.tensor([[1., 2., 3.],      # 3 non-zeros -> int(3*0.5)=1
                      [4., 5., 6.]])     # 3 non-zeros -> int(3*0.5)=1

    denoise_percent = 0.5
    y_noisy, mask = add_noise(y, denoise_percent)

    # Expected zeroed count PER ROW
    for b in range(y.size(0)):
        num_non_zero_row = (y[b] != 0).sum().item()
        expected = int(num_non_zero_row * denoise_percent)

        num_zeroed_row = ((y[b] != 0) & (y_noisy[b] == 0)).sum().item()
        num_masked_row = mask[b].sum().item()

        assert num_zeroed_row == expected, \
            f"Row {b}: expected {expected} values to be zeroed, got {num_zeroed_row}"
        assert num_masked_row == expected, \
            f"Row {b}: expected {expected} masked positions, got {num_masked_row}"


def test_all_zero_input():
    y = torch.zeros(4, 7)

    y_noisy, mask = add_noise(y, denoise_percent=0.9)

    assert torch.equal(y, y_noisy), "All-zero tensor should remain unchanged"
    assert mask.sum().item() == 0, "Mask should be all-False if there are no non-zeros"


if __name__ == "__main__":
    test_no_noise()
    test_only_nonzeros_are_zeroed_and_mask_matches()
    test_correct_number_zeroed_per_sequence()
    test_all_zero_input()
    print("All tests passed.")