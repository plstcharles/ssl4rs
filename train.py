import dotenv
import hydra
import omegaconf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(version_base=None, config_path="configs/", config_name="train.yaml")
def main(config: omegaconf.DictConfig):
    """Main entrypoint for the training pipeline."""
    # imports are nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    import ssl4rs
    import ssl4rs.training_pipeline
    ssl4rs.utils.config.extra_inits(config)
    return ssl4rs.training_pipeline.train(config)


if __name__ == "__main__":
    main()
# import hydra.utils @@@@@ TODO