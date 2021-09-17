from sklearn.model_selection import train_test_split

class BaseDataLoader():
    def __init__(self, data_handler, shuffle, test_split, random_state, stratify, training):
        dh = data_handler

        if dh.X_data_test == dh.y_data_test == None:
            if 0 < test_split < 1:
                stratify = dh.y_data if stratify else None
                X_train, X_test, y_train, y_test = train_test_split(dh.X_data,
                                                                    dh.y_data,
                                                                    test_size=test_split,
                                                                    random_state=random_state,
                                                                    shuffle=shuffle,
                                                                    stratify=stratify)
                self.X_out, self.y_out = (X_train, y_train) if training else (X_test, y_test)
                print("Training and test sets created regarding defined test_split percentage.")
            else:
                self.X_out, self.y_out = dh.X_data, dh.y_data
                print("Whole dataset is used for training.")

        elif dh.X_data_test.all() != None and dh.y_data_test.all() != None:
            self.X_out, self.y_out = (dh.X_data, dh.y_data) if training \
                            else (dh.X_data_test, dh.y_data_test)
            print("For training and testing separate datasets configured in data_handler will be used.")
        else:
            raise ValueError('data_handler not configured properly.')

    def get_data(self):
        print(f"Number of loaded data instances: {len(self.X_out)}")
        return self.X_out, self.y_out



