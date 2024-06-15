from contextlib import contextmanager

import torch

# Determine the device based on availability
if torch.cuda.is_available():
    gpu = torch.device("cuda")
elif torch.backends.mps.is_available():
    gpu = torch.device("mps")
else:
    gpu = torch.device("cpu")

cpu = torch.device("cpu")

# Define high_vram (set it to False by default)
high_vram = True

# No need to test CUDA here; it's handled in the device selection
# torch.zeros((1, 1)).to(gpu, torch.float32)
# torch.cuda.empty_cache()

models_in_gpu = []


@contextmanager
def movable_bnb_model(m):
    if hasattr(m, 'quantization_method'):
        m.quantization_method_backup = m.quantization_method
        del m.quantization_method
    try:
        yield None
    finally:
        if hasattr(m, 'quantization_method_backup'):
            m.quantization_method = m.quantization_method_backup
            del m.quantization_method_backup
    return


def load_models_to_gpu(models):
    global models_in_gpu

    if not isinstance(models, (tuple, list)):
        models = [models]

    models_to_remain = [m for m in set(models) if m in models_in_gpu]
    models_to_load = [m for m in set(models) if m not in models_in_gpu]
    models_to_unload = [m for m in set(models_in_gpu) if m not in models_to_remain]

    if not high_vram:
        for m in models_to_unload:
            with movable_bnb_model(m):
                m.to(cpu)
            print('Unload to CPU:', m.__class__.__name__)
        models_in_gpu = models_to_remain

    for m in models_to_load:
        with movable_bnb_model(m):
            m.to(gpu)
        print('Load to ', gpu, ':', m.__class__.__name__)

    models_in_gpu = list(set(models_in_gpu + models))
    # Only use empty_cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return


def unload_all_models(extra_models=None):
    global models_in_gpu

    if extra_models is None:
        extra_models = []

    if not isinstance(extra_models, (tuple, list)):
        extra_models = [extra_models]

    models_in_gpu = list(set(models_in_gpu + extra_models))

    return load_models_to_gpu([])