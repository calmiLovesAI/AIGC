from diffusers import EulerDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler, \
    EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler

__all__ = [
    'diffusion_schedulers',
    'get_scheduler_names',
    'get_scheduler',
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


def get_scheduler(scheduler_name):
    if scheduler_name not in diffusion_schedulers:
        print(f"The scheduler {scheduler_name} is not in diffusion_schedulers, DPM++ 2M SDE Karras will be used for the default.")
        scheduler_name = 'DPM++ 2M SDE Karras'
    scheduler = diffusion_schedulers[scheduler_name]
    return scheduler

