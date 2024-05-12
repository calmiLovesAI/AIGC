import sys

from src.pipelines.diffusion.modules.shared_cmd_options import cmd_opts


def realesrgan_models_names():
    import src.pipelines.diffusion.modules.realesrgan_model as realesrgan_model
    return [x.name for x in realesrgan_model.get_realesrgan_models(None)]


def postprocessing_scripts():
    import src.pipelines.diffusion.modules.scripts as m_scripts

    return m_scripts.scripts_postproc.scripts


def sd_vae_items():
    import src.pipelines.diffusion.modules.sd_vae as m_sd_vae

    return ["Automatic", "None"] + list(m_sd_vae.vae_dict)


def refresh_vae_list():
    import src.pipelines.diffusion.modules.sd_vae as m_sd_vae

    m_sd_vae.refresh_vae_list()


def cross_attention_optimizations():
    import src.pipelines.diffusion.modules.sd_hijack as m_sd_hijack

    return ["Automatic"] + [x.title() for x in m_sd_hijack.optimizers] + ["None"]


def sd_unet_items():
    import src.pipelines.diffusion.modules.sd_unet as m_sd_unet

    return ["Automatic"] + [x.label for x in m_sd_unet.unet_options] + ["None"]


def refresh_unet_list():
    import src.pipelines.diffusion.modules.sd_unet as m_sd_unet

    m_sd_unet.list_unets()


def list_checkpoint_tiles(use_short=False):
    import src.pipelines.diffusion.modules.sd_models as m_sd_models
    return m_sd_models.checkpoint_tiles(use_short)


def refresh_checkpoints():
    import src.pipelines.diffusion.modules.sd_models as m_sd_models
    return m_sd_models.list_models()


def list_samplers():
    import src.pipelines.diffusion.modules.sd_samplers as m_sd_samplers
    return m_sd_samplers.all_samplers


def reload_hypernetworks():
    from src.pipelines.diffusion.modules.hypernetworks import hypernetwork
    from src.pipelines.diffusion.modules import shared

    shared.hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)


def get_infotext_names():
    from src.pipelines.diffusion.modules import generation_parameters_copypaste, shared
    res = {}

    for info in shared.opts.data_labels.values():
        if info.infotext:
            res[info.infotext] = 1

    for tab_data in generation_parameters_copypaste.paste_fields.values():
        for _, name in tab_data.get("fields") or []:
            if isinstance(name, str):
                res[name] = 1

    return list(res)


ui_reorder_categories_builtin_items = [
    "prompt",
    "image",
    "inpaint",
    "sampler",
    "accordions",
    "checkboxes",
    "dimensions",
    "cfg",
    "denoising",
    "seed",
    "batch",
    "override_settings",
]


def ui_reorder_categories():
    from modules import scripts

    yield from ui_reorder_categories_builtin_items

    sections = {}
    for script in scripts.scripts_txt2img.scripts + scripts.scripts_img2img.scripts:
        if isinstance(script.section, str) and script.section not in ui_reorder_categories_builtin_items:
            sections[script.section] = 1

    yield from sections

    yield "scripts"


class Shared(sys.modules[__name__].__class__):
    """
    this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than
    at program startup.
    """

    sd_model_val = None

    @property
    def sd_model(self):
        import src.pipelines.diffusion.modules.sd_models as m_sd_models

        return m_sd_models.model_data.get_sd_model()

    @sd_model.setter
    def sd_model(self, value):
        import src.pipelines.diffusion.modules.sd_models as m_sd_models

        m_sd_models.model_data.set_sd_model(value)


sys.modules['src.pipelines.diffusion.modules.shared'].__class__ = Shared
