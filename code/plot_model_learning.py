import matplotlib.pyplot as plt
import pandas as pd

log_history = [
    {
      "epoch": 1.0,
      "eval_accuracy": 0.6330731014904187,
      "eval_f1": 0.3432595573440644,
      "eval_loss": 0.977558434009552,
      "eval_precision": 0.28653006382264024,
      "eval_recall": 0.42799799297541397,
      "eval_runtime": 6.2336,
      "eval_samples_per_second": 8.021,
      "eval_steps_per_second": 4.011,
      "step": 75
    },
    {
      "epoch": 2.0,
      "eval_accuracy": 0.6530242094472045,
      "eval_f1": 0.3977847339272815,
      "eval_loss": 0.9170916676521301,
      "eval_precision": 0.3824074074074074,
      "eval_recall": 0.4144505770195685,
      "eval_runtime": 6.0765,
      "eval_samples_per_second": 8.228,
      "eval_steps_per_second": 4.114,
      "step": 150
    },
    {
      "epoch": 3.0,
      "eval_accuracy": 0.6955287437899219,
      "eval_f1": 0.45354602284975215,
      "eval_loss": 0.8531413078308105,
      "eval_precision": 0.3975812547241119,
      "eval_recall": 0.5278474661314602,
      "eval_runtime": 6.0469,
      "eval_samples_per_second": 8.269,
      "eval_steps_per_second": 4.134,
      "step": 225
    },
    {
      "epoch": 4.0,
      "eval_accuracy": 0.6855926188786373,
      "eval_f1": 0.4684925406368292,
      "eval_loss": 0.8951283097267151,
      "eval_precision": 0.4211369095276221,
      "eval_recall": 0.5278474661314602,
      "eval_runtime": 6.0821,
      "eval_samples_per_second": 8.221,
      "eval_steps_per_second": 4.11,
      "step": 300
    },
    {
      "epoch": 5.0,
      "eval_accuracy": 0.7007333806482139,
      "eval_f1": 0.5063904803878361,
      "eval_loss": 0.8827097415924072,
      "eval_precision": 0.45147347740667976,
      "eval_recall": 0.5765178123432012,
      "eval_runtime": 6.0857,
      "eval_samples_per_second": 8.216,
      "eval_steps_per_second": 4.108,
      "step": 375
    },
    {
      "epoch": 6.0,
      "eval_accuracy": 0.7027048340036275,
      "eval_f1": 0.5084145261293179,
      "eval_loss": 0.9082810878753662,
      "eval_precision": 0.45501387237415775,
      "eval_recall": 0.5760160561966884,
      "eval_runtime": 6.159,
      "eval_samples_per_second": 8.118,
      "eval_steps_per_second": 4.059,
      "step": 450
    },
    {
      "epoch": 6.666666666666667,
      "grad_norm": 3.061610698699951,
      "learning_rate": 6.666666666666667e-06,
      "loss": 0.7315,
      "step": 500
    },
    {
      "epoch": 7.0,
      "eval_accuracy": 0.7041242804195252,
      "eval_f1": 0.5208152645273201,
      "eval_loss": 0.9186286926269531,
      "eval_precision": 0.45857197403589156,
      "eval_recall": 0.6026091319618665,
      "eval_runtime": 6.0799,
      "eval_samples_per_second": 8.224,
      "eval_steps_per_second": 4.112,
      "step": 525
    },
    {
      "epoch": 8.0,
      "eval_accuracy": 0.7103540730226323,
      "eval_f1": 0.5372802136657022,
      "eval_loss": 0.9362245202064514,
      "eval_precision": 0.4828,
      "eval_recall": 0.6056196688409433,
      "eval_runtime": 7.1502,
      "eval_samples_per_second": 6.993,
      "eval_steps_per_second": 3.496,
      "step": 600
    },
    {
      "epoch": 9.0,
      "eval_accuracy": 0.7069631732513209,
      "eval_f1": 0.5444369823147527,
      "eval_loss": 0.9729665517807007,
      "eval_precision": 0.49151172190784154,
      "eval_recall": 0.6101354741595585,
      "eval_runtime": 6.4998,
      "eval_samples_per_second": 7.693,
      "eval_steps_per_second": 3.846,
      "step": 675
    },
    {
      "epoch": 10.0,
      "eval_accuracy": 0.709013484740951,
      "eval_f1": 0.5470124013528748,
      "eval_loss": 0.9694036245346069,
      "eval_precision": 0.4967239967239967,
      "eval_recall": 0.60863020572002,
      "eval_runtime": 6.5397,
      "eval_samples_per_second": 7.646,
      "eval_steps_per_second": 3.823,
      "step": 750
    }
  ]

# log_history = [
#     {
#       "epoch": 1.0,
#       "eval_accuracy": 0.9232116788321167,
#       "eval_f1": 0.7407079646017699,
#       "eval_loss": 0.23456259071826935,
#       "eval_precision": 0.7253032928942807,
#       "eval_recall": 0.7567811934900542,
#       "eval_runtime": 2.9143,
#       "eval_samples_per_second": 17.5,
#       "eval_steps_per_second": 8.921,
#       "step": 205
#     },
#     {
#       "epoch": 2.0,
#       "eval_accuracy": 0.9506569343065694,
#       "eval_f1": 0.8153434433541481,
#       "eval_loss": 0.16518202424049377,
#       "eval_precision": 0.8045774647887324,
#       "eval_recall": 0.8264014466546112,
#       "eval_runtime": 2.8486,
#       "eval_samples_per_second": 17.904,
#       "eval_steps_per_second": 9.127,
#       "step": 410
#     },
#     {
#       "epoch": 2.4390243902439024,
#       "grad_norm": 5.06137228012085,
#       "learning_rate": 1.5121951219512196e-05,
#       "loss": 0.332,
#       "step": 500
#     },
#     {
#       "epoch": 3.0,
#       "eval_accuracy": 0.9553284671532847,
#       "eval_f1": 0.8217954443948192,
#       "eval_loss": 0.16292321681976318,
#       "eval_precision": 0.8120035304501324,
#       "eval_recall": 0.8318264014466547,
#       "eval_runtime": 3.6666,
#       "eval_samples_per_second": 13.909,
#       "eval_steps_per_second": 7.091,
#       "step": 615
#     },
#     {
#       "epoch": 4.0,
#       "eval_accuracy": 0.96,
#       "eval_f1": 0.8462572837292693,
#       "eval_loss": 0.16535340249538422,
#       "eval_precision": 0.8391111111111111,
#       "eval_recall": 0.8535262206148282,
#       "eval_runtime": 3.2178,
#       "eval_samples_per_second": 15.849,
#       "eval_steps_per_second": 8.08,
#       "step": 820
#     },
#     {
#       "epoch": 4.878048780487805,
#       "grad_norm": 4.39166259765625,
#       "learning_rate": 1.024390243902439e-05,
#       "loss": 0.0641,
#       "step": 1000
#     },
#     {
#       "epoch": 5.0,
#       "eval_accuracy": 0.9635036496350365,
#       "eval_f1": 0.8611235955056181,
#       "eval_loss": 0.16527867317199707,
#       "eval_precision": 0.8561215370866846,
#       "eval_recall": 0.8661844484629295,
#       "eval_runtime": 3.1596,
#       "eval_samples_per_second": 16.141,
#       "eval_steps_per_second": 8.229,
#       "step": 1025
#     },
#     {
#       "epoch": 6.0,
#       "eval_accuracy": 0.964963503649635,
#       "eval_f1": 0.8648648648648649,
#       "eval_loss": 0.16833718121051788,
#       "eval_precision": 0.8617594254937163,
#       "eval_recall": 0.8679927667269439,
#       "eval_runtime": 2.9264,
#       "eval_samples_per_second": 17.428,
#       "eval_steps_per_second": 8.885,
#       "step": 1230
#     },
#     {
#       "epoch": 7.0,
#       "eval_accuracy": 0.9661313868613138,
#       "eval_f1": 0.8742676881478142,
#       "eval_loss": 0.17923758924007416,
#       "eval_precision": 0.87151841868823,
#       "eval_recall": 0.8770343580470162,
#       "eval_runtime": 2.8904,
#       "eval_samples_per_second": 17.645,
#       "eval_steps_per_second": 8.995,
#       "step": 1435
#     },
#     {
#       "epoch": 7.317073170731708,
#       "grad_norm": 0.07889224588871002,
#       "learning_rate": 5.365853658536586e-06,
#       "loss": 0.0256,
#       "step": 1500
#     },
#     {
#       "epoch": 8.0,
#       "eval_accuracy": 0.9655474452554744,
#       "eval_f1": 0.8669683257918552,
#       "eval_loss": 0.1830425262451172,
#       "eval_precision": 0.8677536231884058,
#       "eval_recall": 0.8661844484629295,
#       "eval_runtime": 2.8889,
#       "eval_samples_per_second": 17.654,
#       "eval_steps_per_second": 9.0,
#       "step": 1640
#     },
#     {
#       "epoch": 9.0,
#       "eval_accuracy": 0.9632116788321168,
#       "eval_f1": 0.8607367475292005,
#       "eval_loss": 0.19023513793945312,
#       "eval_precision": 0.8553571428571428,
#       "eval_recall": 0.8661844484629295,
#       "eval_runtime": 2.8413,
#       "eval_samples_per_second": 17.95,
#       "eval_steps_per_second": 9.151,
#       "step": 1845
#     },
#     {
#       "epoch": 9.75609756097561,
#       "grad_norm": 1.380218267440796,
#       "learning_rate": 4.878048780487805e-07,
#       "loss": 0.0117,
#       "step": 2000
#     },
#     {
#       "epoch": 10.0,
#       "eval_accuracy": 0.9637956204379562,
#       "eval_f1": 0.8634520054078414,
#       "eval_loss": 0.19695580005645752,
#       "eval_precision": 0.8607367475292004,
#       "eval_recall": 0.8661844484629295,
#       "eval_runtime": 3.0484,
#       "eval_samples_per_second": 16.73,
#       "eval_steps_per_second": 8.529,
#       "step": 2050
#     }
#   ]

# df = pd.DataFrame(log_history)
train_size = 150
# df = pd.read_csv(f"/mnt/c/Users/kaska/Downloads/train{train_size}.csv")
df = pd.read_csv(f"/mnt/c/Users/kaska/Downloads/train_full.csv")

# print(df)
# exit(0)

x = df["train/epoch"]
y = df[f"czert_lr2e-05_bs4_train287 - eval/f1"]

plt.plot(x, y)
plt.xlabel("Epoch")
plt.ylabel("F-score")
plt.title(f"train 287 samples")



last_x = x.iloc[-1]
last_y = y.iloc[-1]
plt.annotate(f"{last_y:.3f}", (last_x, last_y), textcoords="offset points", xytext=(0, -10), ha='center')

plt.show()
# df = df.dropna(subset=["eval_f1", "eval_loss"])

# # Create the figure
# fig, ax1 = plt.subplots()

# # Plot F1 score
# ax1.set_xlabel("Epoch")
# ax1.set_ylabel("F1 Score", color="tab:blue")
# ax1.plot(df["epoch"], df["eval_f1"], marker="o", linestyle="-", color="tab:blue", label="F1 Score")
# ax1.tick_params(axis="y", labelcolor="tab:blue")
# ax1.set_ylim(0,1)
# # Create a second y-axis for loss
# # ax2 = ax1.twinx()
# # ax2.set_ylabel("Loss", color="tab:red")
# # ax2.plot(df["epoch"], df["eval_loss"], marker="s", linestyle="--", color="tab:red", label="Loss")
# # ax2.tick_params(axis="y", labelcolor="tab:red")

# # Add a title
# plt.title("FUNSD")
# fig.tight_layout()

# # Show the plot
# plt.show()