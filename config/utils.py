def config_merge_from_file(
    cfg: "yacs.config.CfgNode",
    path_to_config: "Union[str, Path, list[str], list[Path], tuple[str, ...], tuple[Path, ...]]",
) -> "yacs.config.CfgNode":
    if isinstance(path_to_config, (list, tuple)):
        for path_to_config_ in path_to_config:
            cfg.merge_from_file(path_to_config_)
    else:
        cfg.merge_from_file(path_to_config)

    return cfg
