import dotenv
import hydra

try:
    # comet likes to be imported before torch for auto-logging to work
    # as of 2023-03-15 with comet_ml==3.32.4, we get a warning if not
    # (this is likely one of the easiest places to perform the import)
    import comet_ml
except ImportError:
    pass


dotenv.load_dotenv(override=True, verbose=True)


@hydra.main(version_base=None, config_path="ssl4rs/configs/", config_name="ssl4rs-test.yaml")
def main(config):
    """Main entrypoint for the testing pipeline."""
    import ssl4rs  # importing here to avoid delay w/ hydra tab completion

    return ssl4rs.test(config)


if __name__ == "__main__":
    main()
