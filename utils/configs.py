class Config():
    def __init__(self, config):
        self.config_base = config
        self.build_config()

    def build_config(self):
        self.config_dataset = self.config_base["dataset_attributes"]
        self.config_model = self.config_base["model_attributes"]
        self.config_optimizer = self.confi_baseg["optimizer_attributes"]
        self.config_training = self.config_base["training_parameters"]
    