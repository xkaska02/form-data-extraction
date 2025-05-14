# this file is very ugly as it was updated manually for every csv file
import pandas as pd
import matplotlib.pyplot as plt

# # Load CSV
# # csv_path = "wandb_export_2025-05-09T17_30_44.343+02_00.csv" 
# # csv_path = "lilt_roberta_trainsize.csv"
# csv_paths = ["czert.csv", "lilt_xlmroberta.csv", "robeczech.csv", "xlm_roberta.csv"]
# custom_labels = ["czert", "lilt_XLM-RoBERTa","robeczech", "XLM-RoBERTa"]

# # colors = plt.cm.tab10(range(len(csv_paths)))  # 4 unique colors
# color_map = plt.get_cmap("viridis", len(csv_paths))


# for i, path in enumerate(csv_paths):
#     df = pd.read_csv(path)
    
#     # Find the correct eval/f1 column
#     f1_column = [col for col in df.columns if col.endswith(" - eval/f1")]
#     if not f1_column:
#         print(f"No eval/f1 column found in {path}")
#         continue
#     f1_column = f1_column[0]
    
#     # Find step column
#     step_column = [col for col in df.columns if "step" in col.lower()]
#     step_column = step_column[0] if step_column else None

#     # Plot
#     plt.plot(df[step_column], df[f1_column], label=custom_labels[i], color=color_map(i))

# plt.xlabel("Epoch")
# plt.ylabel("F1 Score")
# plt.title("Best model comparison")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.grid(axis="x")
# plt.savefig("best_model_comparison.pdf", format="pdf")
# plt.show()

csv_path = "wandb_data/czert_loss_subtokens.csv"
df = pd.read_csv(csv_path)
out_file = csv_path.replace("wandb_data","graphs")
out_file = csv_path.replace("csv","pdf")
# out_file = "best_model_comparison.pdf"

# df_lilt_robeczech = pd.read_csv(csv_path[1])
# df_lilt_xlmroberta = pd.read_csv(csv_path[2])
# df_robeczech_and_roberta = pd.read_csv(csv_path[3])



# for i in range(1, len(csv_path)):
#     data = pd.read_csv(csv_path[i])
#     dframe = pd.DataFrame(data)
#     df = pd.concat([df, dframe], axis=1)
# Extract the training steps
step = df["train/epoch"]

# print(df.columns)
# exit(0)

# Define the training dataset sizes present in the CSV
# train_sizes = [30, 150, 287]
# head_size = [1,2,3]

# Set up a colormap
# color_map = plt.get_cmap("viridis", len(head_size))
# color_map = plt.get_cmap("viridis", len(train_sizes))
color_map = plt.get_cmap("viridis", 2)
# # Create the plot
# plt.figure(figsize=(10, 6))
# for idx, size in enumerate(train_sizes):
#     column_name = f"robeczech_lr3e-05_bs16_train{size} - eval/f1"
#                     # robeczech_lr3e-05_bs16_train287
#     if column_name in df.columns:
#         plt.plot(step, df[column_name], label=f"train size {size}", color=color_map(idx))
#     else:
#         print("ddpc")
ls = ["True","False"]
for idx, size in enumerate(ls):
    column_name = f"czert_lr2e-05_bs4_train287_label_subtokens_{ls[idx]} - eval/f1"
    if column_name in df.columns:
        plt.plot(step, df[column_name], label=f"{size}", color=color_map(idx))
  
    

# for idx, size in enumerate(train_sizes):
#     column_name = f"lilt_xlmroberta_lr2e-05_bs4_train{size} - eval/f1"
#     if column_name in df.columns:
#         plt.plot(step, df[column_name], label=f"train size {size}", color=color_map(idx))
#     else:
#         column_name = f"lilt-xlm-roberta_lr2e-05_bs4_train{size} - eval/f1"
#         plt.plot(step, df[column_name], label=f"train size {size}", color=color_map(idx))

# plt.plot(step, df[f"czert_lr2e-05_bs4_train287 - eval/f1"], label="czert", color=color_map(0))
# print(df["czert_lr2e-05_bs4_train287 - eval/f1"].isna().sum)
# print(df.columns.tolist())
# exit(0)
# plt.plot(step, df["robeczech_lr3e-05_bs16_train287 - eval/f1"], label="robeczech", color=color_map(1))
# plt.plot(step, df["xlm-roberta_lr3e-05_bs16_train287 - eval/f1"], label="XLM-RoBERTa", color=color_map(2))
# plt.plot(step, df["lilt_robeczech_lr2e-05_bs4_train287 - eval/f1"], label="LiLT_robeczech", color=color_map(3))
# plt.plot(step, df["lilt_xlmroberta_lr2e-05_bs4_train287 - eval/f1"], label="LiLT-XLM-RoBERTa", color=color_map(4))

# Add plot details
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Czert Loss on Subtokens")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

plt.savefig(out_file, format="pdf")
# Show the plot
plt.show()
