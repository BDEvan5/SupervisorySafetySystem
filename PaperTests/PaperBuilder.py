import glob , yaml, csv


class DataBuilder:
    def __init__(self):
        self.data = {}
        self.base_keys = []
        self.n = 0
        self.path = f"Data/PaperVehicles/"

    def build_keys(self):
        with open(f"Data/base_key_builder.yaml") as f:
            key_data = yaml.safe_load(f)

        for key in key_data:
            self.base_keys.append(key)
            # self.data[key] = []

    def read_data(self):
        folders = glob.glob(f"{self.path}*/")
        for i, folder in enumerate(folders):
            print(f"Folder being opened: {folder}")
            
            try:
                config = glob.glob(folder + '/*_record.yaml')[0]
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Filename issue: {folder}")
                continue            

            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                continue

            self.data[i] = {}
            # self.data[i]['folder'] = folder
            for key in config_data.keys():
                if key == "SSS" or key == "Wo":
                    for sub_key in config_data[key].keys():
                        # if sub_key in self.base_keys:
                        store_key = f"{key}_{sub_key}"
                        self.data[i][store_key] = config_data[key][sub_key]
                        if not store_key in self.base_keys:
                            self.base_keys.append(store_key)
                    continue
                if key in self.base_keys:
                    self.data[i][key] = config_data[key]


    def save_data_table(self, name="DataTable"):
        directory = "PaperProcessing/" + name + ".csv"
        with open(directory, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=self.base_keys)
            writer.writeheader()
            for key in self.data.keys():
            # for i in range(len(self.data.keys())):
                writer.writerow(self.data[key])


        print(f"Data saved to {name} --> {len(self.data)} Entries")

def run_builder():
    db = DataBuilder()
    db.build_keys()
    db.read_data()
    db.save_data_table()

if __name__ == "__main__":
    run_builder()


