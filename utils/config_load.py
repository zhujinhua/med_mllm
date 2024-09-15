import yaml

def load_yaml_file(file_path):
    """
    Load YAML file into a dictionary
    :param file_path: Path to the YAML file
    """
    with open(file_path, 'r',encoding="utf8") as file:
        return yaml.safe_load(file)


# with open("./data/test.yaml", encoding="utf-8") as yaml_file:
#     data = yaml.safe_load(yaml_file)

if __name__ == "__main__":
    print(load_yaml_file(file_path="./data/test.yaml"))