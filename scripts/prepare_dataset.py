from pathlib import Path

import pandas as pd
import typer
from rich import print
from sklearn.model_selection import StratifiedShuffleSplit


RANDOM_STATE = 42


def main(input_filepath: Path, output_dir: Path, test_size: float = 0.1):
    data_df = pd.read_csv(input_filepath)

    print(f"[yellow]🧐 Total samples: {data_df.shape[0]}[/yellow]")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)

    data_df["group"] = pd.cut(data_df["target"], bins=5, labels=[f"g{i}" for i in range(5)])
    for train_index, test_index in sss.split(data_df, data_df[["group"]]):
        sss_train = data_df.iloc[train_index]
        sss_test = data_df.iloc[test_index]

    sss_train.to_csv(output_dir.joinpath("train.csv"), index=False)
    sss_test.to_csv(output_dir.joinpath("validation.csv"), index=False)

    print("[green]✅ Success saved.[/green]")


if __name__ == "__main__":
    typer.run(main)
