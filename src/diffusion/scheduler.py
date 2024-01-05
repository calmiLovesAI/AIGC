from diffusers import EulerDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler, \
    EulerAncestralDiscreteScheduler

__all__ = [
    'diffusion_schedulers'
]

diffusion_schedulers = {
    'pndm': PNDMScheduler,
    'euler_discrete': EulerDiscreteScheduler,
    'dpm_solver_multistep': DPMSolverMultistepScheduler,
    # a1111 schedulers
    'DPM++ 2M': DPMSolverMultistepScheduler,
    'DPM++ 2M Karras': DPMSolverMultistepScheduler(use_karras_sigmas=True),
    'DPM++ 2M SDE': DPMSolverMultistepScheduler(algorithm_type='sde-dpmsolver++'),
    'DPM++ 2M SDE Karras': DPMSolverMultistepScheduler(algorithm_type='sde-dpmsolver++',
                                                       use_karras_sigmas=True),
    'Euler': EulerDiscreteScheduler,
    'Euler a': EulerAncestralDiscreteScheduler,
}
