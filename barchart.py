import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu
models = ['Random Forest', 'Catboost', 'LightGBM', 'XGBoost']
precision = [0.73, 0.52, 0.73, 0.82]
recall = [0.76, 0.82, 0.58, 0.70]
f1_score = [0.75, 0.64, 0.65, 0.76]
accuracy = [0.92, 0.87, 0.91, 0.94]

# Tạo biểu đồ
x = np.arange(len(models))  # vị trí của các nhóm
width = 0.2  # chiều rộng của các thanh

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, precision, width, label='Precision')
rects2 = ax.bar(x - 0.5*width, recall, width, label='Recall')
rects3 = ax.bar(x + 0.5*width, f1_score, width, label='F1-Score')
rects4 = ax.bar(x + 1.5*width, accuracy, width, label='Accuracy')

# Thêm các chi tiết
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Metrics')
ax.set_xticks(x)
ax.set_xticklabels(models)

# Move the legend to the bottom center
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

# Hiển thị giá trị trên các thanh
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.show()