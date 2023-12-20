from diffusers import EulerDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler

__all__ = [
    'diffusion_schedulers'
]

diffusion_schedulers = {
    'pndm': PNDMScheduler,
    'euler_discrete': EulerDiscreteScheduler,
    'dpm_solver_multistep': DPMSolverMultistepScheduler
}