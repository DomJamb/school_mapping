import os
import argparse
import pandas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--exp_name", help="Config file", default="all_convnext_large")
    parser.add_argument("--iso", help="ISO code", default=[
        'ATG', 'AIA', 'YEM', 'SEN', 'BWA', 'MDG', 'BEN', 'BIH', 'BLZ', 'BRB', 
        'CRI', 'DMA', 'GHA', 'GIN', 'GRD', 'HND', 'HUN', 'KAZ', 'KEN', 'KIR', 
        'KNA', 'LCA', 'MNG', 'MSR', 'MWI', 'NAM', 'NER', 'NGA', 'PAN', 'RWA', 
        'SLE', 'SLV', 'SSD', 'THA', 'TTO', 'UKR', 'UZB', 'VCT', 'VGB', 'ZAF', 
        'ZWE', 'BRA'
    ], nargs='+')

    args = parser.parse_args()

    cwd = os.path.dirname(os.getcwd())
    cwd = "/home/agorup/school_mapping/"
    dir = os.path.join(cwd, "exp", args.exp_name)
    iso_codes = args.iso

    worst_school_prob = 0
    worst_school_example = ""
    worst_non_school_prob = 0
    worst_non_school_example = ""

    for iso_code in iso_codes:
        iso_path = os.path.join(dir, iso_code)
        if os.path.exists(iso_path):
            csv_file = pandas.read_csv(os.path.join(iso_path, f"{iso_code}.csv"))

            for i in csv_file.index:
                example = csv_file["UID"][i]
                
                if csv_file["y_true"][i] != csv_file["y_preds"][i]:

                    prob = csv_file["y_probs"][i]
                    if prob < worst_non_school_prob and prob < worst_school_prob:
                        continue

                    if "NON_SCHOOL" in example: 
                        worst_non_school_prob = prob
                        worst_non_school_example = example

                    else:
                        worst_school_prob = prob
                        worst_school_example = example

    print(f"Worst school example:{worst_school_example}")
    print(f"Worst school prob: {worst_school_prob}")
    print(f"Worst non-school example:{worst_non_school_example}")
    print(f"Worst non-school prob: {worst_non_school_prob}")
            




