import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):
    """Main entrypoint for the testing pipeline."""
    # imports are nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    import ssl4rs.utils
    import ssl4rs.testing_pipeline
    ssl4rs.utils.config.extra_inits(config)
    return ssl4rs.testing_pipeline.test(config)


if __name__ == "__main__":
    main()
