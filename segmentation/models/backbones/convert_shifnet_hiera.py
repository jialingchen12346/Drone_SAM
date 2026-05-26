"""
Extract SHIFNet's FMB-trained Hiera backbone weights for Drone-SAM-Adapter.

The remapping strips SHIFNet's prefix and drops SHIFNet-specific adapters,
producing a state_dict loadable directly into a raw Hiera trunk (before
BottleneckAdapter wrapping).

SHIFNet keys (fmb.pth):      image_encoder.trunk.blocks.{N}.attn.qkv.weight
Output keys (raw Hiera):     blocks.{N}.attn.qkv.weight

Note: the '.block.' nesting appears later, after BottleneckAdapter wrapping
in SAM2HieraAdapter.__init__. The injection point is between
  self.trunk = model.image_encoder.trunk  (line 80)
and
  self.trunk.blocks = adapted_blocks      (line 92)

Usage (CLI):
    python convert_shifnet_hiera.py /path/to/fmb.pth [output.pth]

Usage (API):
    sd = extract_and_remap_hiera_weights('/path/to/fmb.pth')
    torch.save(sd, 'shifnet_fmb_hiera.pth')
"""

import re
import torch


PREFIX = 'image_encoder.trunk.'


def _remap_key(key: str) -> str | None:
    """Remap one SHIFNet key to raw Hiera key, or return None to skip."""
    if 'Space_Adapter' in key or 'MLP_Adapter' in key:
        return None
    if not key.startswith(PREFIX):
        return None

    # Strip 'image_encoder.trunk.' prefix → raw Hiera key
    # e.g. image_encoder.trunk.blocks.0.norm1.weight → blocks.0.norm1.weight
    return key[len(PREFIX):]


def extract_and_remap_hiera_weights(shifnet_path: str) -> dict:
    """Load SHIFNet fmb.pth, extract standard Hiera keys, strip prefix.

    Returns a state_dict (586 keys) with keys like 'trunk.blocks.0.norm1.weight',
    loadable into a raw Hiera via load_state_dict(sd, strict=False).
    The SHIFNet-specific Space_Adapter/MLP_Adapter keys are skipped.
    """
    sd = torch.load(shifnet_path, map_location='cpu', weights_only=False)
    new_sd: dict[str, torch.Tensor] = {}
    skipped = 0
    for old_key, tensor in sd.items():
        new_key = _remap_key(old_key)
        if new_key is None:
            skipped += 1
            continue
        new_sd[new_key] = tensor

    print(f'Remapped: {len(new_sd)} keys kept, {skipped} skipped (of {len(sd)} total)')
    return new_sd


def verify_mapping(shifnet_path: str) -> bool:
    """Verify that remapped keys match a raw Hiera trunk (pre-BottleneckAdapter)."""
    sd = extract_and_remap_hiera_weights(shifnet_path)

    import sys
    sys.path.insert(0, '/home/jl/Drone-SAM-Adapter')
    from sam2.modeling.backbones.hieradet import Hiera

    # We need the raw Hiera trunk before wrapping.
    # Build it the same way SAM2HieraAdapter does, but stop before wrapping.
    from sam2.build_sam import build_sam2
    model = build_sam2(
        'configs/sam2.1/sam2.1_hiera_l.yaml',
        '/home/jl/Drone-SAM-Adapter/checkpoints/sam2.1_hiera_large.pt',
        device='cpu',
    )
    raw_trunk = model.image_encoder.trunk
    trunk_keys = set(raw_trunk.state_dict().keys())

    source_keys = set(sd.keys())
    extra_in_src = source_keys - trunk_keys
    missing_in_src = trunk_keys - source_keys

    if missing_in_src:
        print(f'  MISSING in remapped (trunk expects): {len(missing_in_src)}')
        for k in sorted(missing_in_src)[:10]:
            print(f'    {k}')
    if extra_in_src:
        print(f'  EXTRA in remapped (not in trunk): {len(extra_in_src)}')
        for k in sorted(extra_in_src)[:10]:
            print(f'    {k}')

    if not missing_in_src and not extra_in_src:
        print('  VERIFIED: All 586 keys match raw Hiera trunk exactly.')
        return True
    else:
        print(f'  WARNING: {len(missing_in_src)} missing, {len(extra_in_src)} extra.')
        return False


def test_load(shifnet_path: str) -> bool:
    """Load remapped weights into raw Hiera trunk, verify weights actually change
    from the default SAM2 init values."""
    import sys
    sys.path.insert(0, '/home/jl/Drone-SAM-Adapter')
    from sam2.build_sam import build_sam2

    remapped = extract_and_remap_hiera_weights(shifnet_path)

    # Build raw SAM2 Hiera trunk (same as SAM2HieraAdapter.__init__ line 79-80)
    model = build_sam2(
        'configs/sam2.1/sam2.1_hiera_l.yaml',
        '/home/jl/Drone-SAM-Adapter/checkpoints/sam2.1_hiera_large.pt',
        device='cpu',
    )
    trunk = model.image_encoder.trunk

    # Compare a tracked key before/after loading SHIFNet weights
    key_check = 'blocks.0.norm1.weight'
    before = trunk.state_dict()[key_check].clone()

    # Load SHIFNet weights into raw trunk (pre-wrapping)
    missing, unexpected = trunk.load_state_dict(remapped, strict=False)
    print(f'  Load: {len(missing)} missing, {len(unexpected)} unexpected')

    after = trunk.state_dict()[key_check]
    changed = not torch.equal(before, after)
    print(f'  {key_check}: weights changed? {changed}')

    max_diff = (after.float() - before.float()).abs().max().item()
    print(f'  max |Δ| = {max_diff:.6f}')
    print(f'  before mean={before.float().mean():.6f}, after mean={after.float().mean():.6f}')

    if missing:
        print(f'  Missing keys (expect 192 Space_Adapter + 192 MLP_Adapter in trunk): {len(missing)}')
        for k in sorted(missing)[:5]:
            print(f'    {k}')
    if unexpected:
        print(f'  Unexpected keys: {len(unexpected)}')

    return changed


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert SHIFNet Hiera weights for Drone-SAM-Adapter')
    parser.add_argument('shifnet_path', help='Path to SHIFNet fmb.pth')
    parser.add_argument('output_path', nargs='?',
                        default='/home/jl/Drone-SAM-Adapter/checkpoints/shifnet_fmb_hiera.pth')
    parser.add_argument('--verify', action='store_true', help='Run key-mapping verification')
    parser.add_argument('--test', action='store_true', help='Test loading into model')
    args = parser.parse_args()

    if args.verify:
        verify_mapping(args.shifnet_path)
    elif args.test:
        ok = test_load(args.shifnet_path)
        print(f'\nTest result: {"PASS" if ok else "FAIL"}')
    else:
        remapped = extract_and_remap_hiera_weights(args.shifnet_path)
        torch.save(remapped, args.output_path)
        print(f'Saved → {args.output_path}')
