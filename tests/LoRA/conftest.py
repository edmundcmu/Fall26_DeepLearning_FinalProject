import pytest
import torch


@pytest.fixture
def randomize_b():
    """Give the B matrices of some groups random values, so the adapters change the output."""

    def randomize(lora, groups, std=0.1):
        with torch.no_grad():
            for p in (p for g in groups for p in lora.lora_b_parameters(g)):
                p.normal_(std=std)

    return randomize
