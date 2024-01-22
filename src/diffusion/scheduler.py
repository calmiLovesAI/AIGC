from diffusers import EulerDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler, \
    EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler

__all__ = [
    'diffusion_schedulers',
    'get_scheduler_names',
]

diffusion_schedulers = {
    # a1111 schedulers
    'DPM++ 2M': DPMSolverMultistepScheduler,
    'DPM++ 2M Karras': DPMSolverMultistepScheduler(use_karras_sigmas=True),
    'DPM++ 2M SDE': DPMSolverMultistepScheduler(algorithm_type='sde-dpmsolver++'),
    'DPM++ 2M SDE Karras': DPMSolverMultistepScheduler(algorithm_type='sde-dpmsolver++',
                                                       use_karras_sigmas=True),
    'Euler': EulerDiscreteScheduler,
    'Euler a': EulerAncestralDiscreteScheduler,
    'DPM++ SDE Karras': DPMSolverSinglestepScheduler(use_karras_sigmas=True),
}


def get_scheduler_names():
    return list(diffusion_schedulers.keys())
