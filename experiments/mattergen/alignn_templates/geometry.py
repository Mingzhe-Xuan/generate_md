"""Differentiable periodic edges and line-graph cosines with fixed connectivity."""
import torch


def recover_images(native_vectors, cell, frac, src, dst):
    images=torch.round(native_vectors@torch.linalg.inv(cell)-(frac[dst]-frac[src]))
    restored=(frac[dst]-frac[src]+images)@cell
    if not torch.allclose(restored,native_vectors,atol=2e-5,rtol=2e-5):
        raise ValueError('Could not reconstruct signed periodic edge images.')
    return images.detach()


def edge_features(cell, frac, src, dst, images, line_src, line_dst):
    vectors=(frac[dst]-frac[src]+images)@cell
    lengths=torch.linalg.vector_norm(vectors,dim=1)
    if not torch.isfinite(lengths).all() or torch.any(lengths<1e-6):
        raise ValueError('Degenerate ALIGNN edge geometry.')
    first,second=-vectors[line_src],vectors[line_dst]
    cosines=(first*second).sum(-1)/(torch.linalg.vector_norm(first,dim=1)*torch.linalg.vector_norm(second,dim=1))
    return lengths,cosines.clamp(-1,1)
