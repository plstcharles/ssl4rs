import dotenv
import hydra

dotenv.load_dotenv(override=True)


@hydra.main(version_base=None, config_path="configs/", config_name="test.yaml")
def main(config):
    """Main entrypoint for the testing pipeline."""
    import ssl4rs  # importing here to avoid delay w/ hydra tab completion
    return ssl4rs.test(config)


if __name__ == "__main__":
    main()
