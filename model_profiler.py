import dotenv
import hydra

dotenv.load_dotenv(override=True, verbose=True)


@hydra.main(version_base=None, config_path="configs/", config_name="profiler.yaml")
def main(config):
    """Main entrypoint for the model profiler pipeline."""
    import ssl4rs  # importing here to avoid delay w/ hydra tab completion

    return ssl4rs.model_profiler(config)


if __name__ == "__main__":
    main()
