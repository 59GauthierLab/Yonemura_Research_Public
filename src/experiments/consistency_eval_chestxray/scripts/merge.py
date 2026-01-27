import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# model,correlation,n_samples
# model1,0.3724453753782449,20
# model2,-0.5131703743207595,20
# model3,-0.4669017201464005,20
# model4,-0.44935034813415603,20
# model5,-0.3161523560939248,20

DIR = Path(__file__).resolve().parent

def merge_csv_files(file_list: list[Path]) -> pd.DataFrame:
    dataframes: list[pd.DataFrame] = [pd.read_csv(file) for file in file_list]
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df

def mean_by_model(df: pd.DataFrame) -> pd.DataFrame:
    mean_df = df.groupby("model", as_index=False).mean()
    return mean_df

def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    input_files = [result_dir / "modelwise_correlation.csv" for result_dir in DIR.parent.iterdir() if (result_dir / "modelwise_correlation.csv").exists()]
    output_dir = DIR / "merged_results.csv"
    merged_df = merge_csv_files(input_files)
    mean_df = mean_by_model(merged_df)
    write_csv(mean_df, output_dir)
    print(f"Merged results written to {output_dir}")

if __name__ == "__main__":
    main()
