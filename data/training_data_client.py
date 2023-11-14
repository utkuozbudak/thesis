import numpy as np
from sklearn.model_selection import train_test_split


class TrainingDataClient:
    @staticmethod
    def load_training_data(dataset: str = "burgers", test_size: float = 0.2, verbose: bool = True):
        """
        Prepares training and testing data for a specified dataset.

        Parameters:
        dataset (str): The name of the dataset to be loaded. Default is 'burgers'.
        test_size (float): The proportion of the dataset to include in the test split.

        Returns:
        tuple: A tuple containing five elements - u0_train, u0_test, u1_train, u1_test, grid.
               These represent the training and testing splits of the initial conditions
               and solutions, and grid, respectively.
        """
        if dataset == "burgers":
            X = np.load("../data/burgers/u0.npy")  # initial conditions
            y = np.load("../data/burgers/u1.npy")  # solutions

            # Split data using train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=42
            )

            grid = np.linspace(0, 2 * np.pi, 256).reshape(-1, 1)
            
            if verbose:
                # Print shapes
                print(f"X_train shape: {X_train.shape}")
                print(f"X_test shape: {X_test.shape}")
                print(f"y_train shape: {y_train.shape}")
                print(f"y_test shape: {y_test.shape}")
                print(f"grid shape: {grid.shape}")

            return X_train, X_test, y_train, y_test, grid
