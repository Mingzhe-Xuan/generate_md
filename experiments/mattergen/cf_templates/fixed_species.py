"""Use exact immutable per-site one-hot elements while retaining geometry gradients."""
import torch


def forward(author, *, optimization_targets, fixed_inputs, scalers, temperature,
            bandgap_model, e_form_model, adding_noise_scale, limit_coords_displacement,
            abc_range, angle_range):
    lengths, angles, coords, elements, _ = optimization_targets
    onehot = fixed_inputs['fixed_atomic_distribution']
    if not torch.equal(elements, onehot):
        raise ValueError('Fixed original per-site elements changed.')
    values = author.Crystalformer_preprocess(
        scaled_batch_abc=lengths, scaled_batch_angle=angles, batch_dir_coords=coords,
        atomic_distribution=elements, init_coords=fixed_inputs['init_coords'], size=fixed_inputs['size'],
        temperature=temperature, abc_scaler=scalers['abc_scaler'], angle_scaler=scalers['angle_scaler'],
        atomic_mask=fixed_inputs['atomic_mask'], ox_states_used_mask=fixed_inputs['ox_states_used_mask'],
        ox_mask_learnable_tensor_per_crystal=None, adding_noise_scale=adding_noise_scale,
        limit_coords_displacement=limit_coords_displacement, abc_range=abc_range, angle_range=angle_range)
    abc, angle, frac, _, pos, lattice, ox = values
    gap, ef = author.Crystalformer_prediction(
        bandgap_model=bandgap_model, e_form_model=e_form_model, normalized_dist=onehot,
        pos=pos, batch=fixed_inputs['batch'], trans_vec=lattice, size=fixed_inputs['size'])
    return gap, ef, dict(bandgap_pred=gap, ef_pred=ef, batch_abc=abc, batch_angle=angle,
        lattice_vectors=lattice, pos=pos, normalized_dist=onehot, normed_batch_dir_coords=frac,
        batch_dir_coords=coords, sharpened_ox_mask=ox, batch=fixed_inputs['batch'])
